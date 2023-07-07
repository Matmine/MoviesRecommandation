from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.functions import mean



class RecommendationEngine:
    
    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        if user_id <= self.max_user_identifier :
            return True
        else:
            return False
    
    def create_user(self, user_id):
        if user_id!= None :
            if user_id.isnumeric():
                if RecommendationEngine.is_user_known(self, user_id) :
                    return user_id
                elif user_id > self.max_user_identifier :
                    self.max_user_identifier = user_id
                    return user_id
                else :
                    user_id = self.max_user_identifier + 1
                    return user_id
        else :
            user_id = self.max_user_identifier + 1
            self.max_user_identifier = user_id
            return user_id


    def get_movie(self, movie_id):
        if movie_id != None :
            filteredMovieDf = self.movies_df.filter(self.movies_df.movieId == movie_id)
            if filteredMovieDf :
                return filteredMovieDf
            print("The movie with id :" + movie_id + " have not been found")     
            return self.best_movies_df.orderBy(rand()).limit(1)               

    def get_ratings_for_user(self, user_id):
    # Méthode pour obtenir les évaluations d'un utilisateur
        if user_id is not None:
            ratingsForUser = self.ratings_df.filter(self.ratings_df.userId == user_id)
            return ratingsForUser.select("movieId", "userId", "rating")


    def add_ratings(self, user_id, ratings):
        if isinstance(ratings, pyspark.sql.DataFrame):
            # Convert Spark DataFrame to a list
            ratings = ratings.select("rating").rdd.flatMap(lambda x: x).collect()
        # Méthode pour ajouter de nouvelles évaluations et ré-entraîner le modèle
        new_ratings_df = self.sc.createDataFrame(ratings, ["userId", "movieId", "rating"])
        self.ratings_df = self.ratings_df.union(new_ratings_df)

        # Diviser les données en ensembles d'entraînement et de test
        self.training_data, self.test_data = self.ratings_df.randomSplit([0.8, 0.2])

        # Ré-entraînement du modèle
        self.__train_model()

        # Retourner une indication de succès ou toute autre information pertinente
        return None


    def predict_rating(self, user_id, movie_id):
        # Méthode pour prédire une évaluation pour un utilisateur et un film donnés
        rating_df = self.sc.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
        predictions = self.model.transform(rating_df)

        if predictions.isEmpty():
            return -1
        else:
            prediction = predictions.select("prediction").collect()[0][0]
            return prediction


    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        user_df = self.sc.createDataFrame([(user_id,)], ["userId"])
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)

        recommended_movies = recommendations.select("userId", explode("recommendations").alias("movies"))
        recommended_movies = recommended_movies.select("userId", col("movies.movieId").alias("movieId"))

        recommended_movies_with_details = recommended_movies.join(self.movies_df, on=["movieId"], how="inner")

        return recommended_movies_with_details.select("title", "genres", ...)


    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec l'algorithme ALS
        als = ALS(maxIter=self.maxIter, regParam=self.regParam, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = als.fit(self.training_data)

        # Évaluation du modèle
        self.__evaluate()


    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne (RMSE)
        predictions = self.model.transform(self.test_data)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)

        self.rmse = rmse
        print(f"RMSE: {rmse}")


    
        # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
        

    def __init__(self, sc, movies_set_path, ratings_set_path, regParam, maxIter, rmse, model):
        self.sparkSession = sc
        self.sqlContext = SQLContext(sc)

        movies_df = self.sparkSession.read.csv(movies_set_path, header=True, inferSchema=True)
        ratings_df = self.sparkSession.read.csv(ratings_set_path, header=True, inferSchema=True)

        movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True)
        ])
        ratings_schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            StructField("timestamp", IntegerType(), True)
        ])

        self.movies_df = self.sqlContext.createDataFrame(movies_df.rdd, movies_schema)
        self.ratings_df = self.sqlContext.createDataFrame(ratings_df.rdd, ratings_schema)
        

        # Join movies_df and ratings_df on the movieId column
        joined_df = movies_df.join(ratings_df, on="movieId")
        # Calculate the mean rating for each movie
        mean_ratings_df = joined_df.groupBy("movieId", "title").agg(mean("rating").alias("meanRating"))
        # Rank the movies based on meanRating in descending order
        ranked_movies_df = mean_ratings_df.orderBy("meanRating", ascending=False)
        # Select the top 100 movies
        top_100_movies_df = ranked_movies_df.limit(100)
        # Select the columns "title" and "meanRating"
        self.best_movies_df = top_100_movies_df.select("title", "meanRating")
          
        
        self.max_user_identifier = self.ratings_df.selectExpr("max(userId)").first()[0]
        self.sc = sc
        self.regParam = regParam
        self.maxIter = maxIter
        self.rmse = rmse
        self.model = model
        self.training_data, self.test_data = self.ratings_df.randomSplit([0.8, 0.2])

        self.__train_model()

    




# Création d'une instance de la classe RecommendationEngine
sc = SparkSession.builder.getOrCreate()
engine = RecommendationEngine(sc, "/workspaces/MoviesRecommandation/app/ml-latest/movies.csv", "/workspaces/MoviesRecommandation/app/ml-latest/ratings.csv", 0.01, 5, None, None)

# Exemple d'utilisation des méthodes de la classe RecommendationEngine
user_id = engine.create_user(None)
print("||||||||||||||||||| USER_ID =" + str(user_id) + "|||||||||||||||||||")
if engine.is_user_known(200):
    movie = engine.get_movie(None)
    print("||||||||||||||||||| MOVIE =" + str(movie) + "|||||||||||||||||||")
    
    ratings = engine.get_ratings_for_user(200)
    print("||||||||||||||||||| RATINGS =" + str(ratings) + "|||||||||||||||||||")
    
    engine.add_ratings(user_id, ratings)
    
    prediction = engine.predict_rating(user_id, movie.movieId)
    
    recommendations = engine.recommend_for_user(user_id, 10)

