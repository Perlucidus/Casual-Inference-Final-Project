import math
import numpy as np
import pandas as pd
from scipy import special
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class MFModel:
    def __init__(self):
        self.ratings = None
        self.users = None
        self.items = None
        self.num_features = None

    def fit(self, data, **kwargs):
        """
        Fits the model on the data
        :param data:
         Tuple(ratings, users, items) where ratings is a dictionary {(user, item): rating},
         users are a dictionary {user: list of items the user has been exposed to},
         items are a dictionary {item: list of users that had been exposed to the item}
        :param kwargs: Optional arguments
        num_features - Number of latent variables
        """
        self.num_features = int(kwargs.get('num_features', 30))
        self.ratings, self.users, self.items = data

    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        raise NotImplementedError()

    def evaluate(self, data):
        """
        Computes RMSE and MAE metrics for the data
        :param data:
         Tuple(ratings, users, items) where ratings is a dictionary {(user, item): rating},
         users are a dictionary {user: list of items the user has been exposed to},
         items are a dictionary {item: list of users that had been exposed to the item}
        :return: dictionary {(metric: value)}
        """
        ratings, users, items = data
        ratings = {(u, i): r for (u, i), r in ratings.items() if u in self.users and i in self.items}
        return {
            'RMSE': math.sqrt(sum(np.square(self.predict(u, m) - r) for (u, m), r in ratings.items()) / len(ratings)),
            'MAE': sum(np.abs(self.predict(u, m) - r) for (u, m), r in ratings.items()) / len(ratings)
        }


class GaussianMF(MFModel):
    def __init__(self):
        super().__init__()
        self.user_preferences = None
        self.item_attributes = None
        self.genres = None
        self.propensity = None

    def propensity_score(self, exposure):
        """
        Computes the propensity scores for users to be exposed to an item
        :param exposure: the exposure model
        """
        self.propensity = {}
        if exposure == 'Bernoulli':
            self.propensity = {(u, i): len(self.items[i]) / len(self.users) for u in self.users for i in self.items}
        elif exposure == 'Covariates':
            for u, exposed in tqdm(self.users.items(), desc='Computing propensity scores'):
                preferences = dict()  # dictionary {genre: probability of user to see a movie of that genre}
                for i in exposed:  # All movies that the user has rated
                    for genre in self.genres[i]:  # All genres of the movie
                        if genre not in preferences:
                            preferences[genre] = 0
                        preferences[genre] += self.ratings[u, i]  # Sum all ratings of this user towards this genre
                total_ratings = sum(r for (user, _), r in self.ratings.items())  # Sum of all ratings of this user
                preferences = {genre: ratings / total_ratings for genre, ratings in preferences.items()}  # Normalize
                for i in self.items:
                    # Sum probabilities for each genre of the movie
                    self.propensity[u, i] = sum(preferences[genre] for genre in self.genres[i] if genre in preferences)
        elif exposure == 'Uniform':
            self.propensity = {(u, i): 1 / (len(self.users) * len(self.items)) for u in self.users for i in self.items}
        else:
            raise Exception('Invalid method for computing propensity scores')

    def loss(self, lambda_u, lambda_i):
        """
        The loss function for Gaussian MF
        :param lambda_u: Inverse standard deviation for user preferences
        :param lambda_i: Inverse standard deviation for item attributes
        :return:
        """
        return sum((self.ratings[u, i] - self.predict(u, i)) ** 2 / self.propensity[u, i]
                   for (u, i), r in self.ratings.items()) \
               + lambda_u * sum(np.linalg.norm(self.user_preferences[u]) ** 2 for u in self.users) \
               + lambda_i * sum(np.linalg.norm(self.item_attributes[i]) ** 2 for i in self.items)

    def gaussian_matrix_factorization(self, lambda_u, lambda_i, num_iterations):
        """
        Performs matrix factorization to find user preferences and item attributes
        :param lambda_u: Inverse standard deviation for user preferences
        :param lambda_i: Inverse standard deviation for item attributes
        :param num_iterations: Number of iterations to perform
        :return: User and item intrinsic scores (user preferences and item attributes)
        """
        # Randomly initialize intrinsic scores
        theta_u = {u: np.random.rand(self.num_features) for u in self.users}
        theta_i = {i: np.random.rand(self.num_features) for i in self.items}
        # Pre-compute outer product
        theta_i_outer = {i: np.outer(theta_i[i], theta_i[i]) for i in self.items}
        with tqdm(total=num_iterations) as progress:
            for t in range(num_iterations):
                progress.set_description(f'GMF Iteration {t + 1}/{num_iterations}')
                for u in self.users:
                    x = sum(theta_i_outer[i] / self.propensity[u, i] for i in self.users[u]) \
                        + lambda_i * np.eye(self.num_features)
                    theta_u[u] = np.linalg.inv(x) \
                                 @ sum(self.ratings[u, i] * theta_i[i] / self.propensity[u, i] for i in self.users[u])
                # Pre-compute outer product
                theta_u_outer = {u: np.outer(theta_u[u], theta_u[u]) for u in self.users}
                for i in self.items:
                    x = sum(theta_u_outer[u] / self.propensity[u, i] for u in self.items[i]) \
                        + lambda_u * np.eye(self.num_features)
                    theta_i[i] = np.linalg.inv(x) \
                                 @ sum(self.ratings[u, i] * theta_u[u] / self.propensity[u, i] for u in self.items[i])
                # Pre-compute outer product
                theta_i_outer = {i: np.outer(theta_i[i], theta_i[i]) for i in self.items}
                self.user_preferences, self.item_attributes = theta_u, theta_i  # Update intrinsic scores
                progress.set_postfix_str(f'Loss: {self.loss(lambda_u, lambda_i)}')
                progress.update()
        return theta_u, theta_i

    def fit(self, data, **kwargs):
        """
        Fits the model on the data
        :param data:
         Tuple(ratings, users, items) where ratings is a dictionary {(user, item): rating},
         users are a dictionary {user: list of items the user has been exposed to},
         items are a dictionary {item: list of users that had been exposed to the item}
        :param kwargs:
        num_features - Number of latent variables
        lambda_u - Inverse standard deviation for user preferences
        lambda_i - Inverse standard deviation for item attributes
        num_iterations - Number of iterations to perform
        exposure - The exposure model
        genres - Dictionary {movie: list of genres for that movie}
        :return:
        """
        super().fit(data, **kwargs)
        lambda_u = float(kwargs.get('lambda_u', 1))
        lambda_i = float(kwargs.get('lambda_i', 1))
        num_iterations = int(kwargs.get('num_iterations', 10))
        exposure = kwargs.get('exposure', 'Bernoulli')
        self.genres = kwargs.get('genres', None)
        self.propensity_score(exposure)  # Compute propensity scores for the MF using the chosen exposure model
        theta = self.gaussian_matrix_factorization(lambda_u, lambda_i, num_iterations)
        self.user_preferences, self.item_attributes = theta

    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        rating = self.user_preferences[user] @ self.item_attributes[item]
        # Fit to our rating range
        rating = max(rating, 0)
        rating = min(rating, 5)
        # rating = round(rating * 2) / 2
        return rating


class PoissonMF(MFModel):
    def __init__(self):
        super().__init__()
        self.user_preferences_shape = None
        self.user_preferences_rate = None
        self.item_attributes_shape = None
        self.item_attributes_rate = None

    def fit(self, data, **kwargs):
        """
        Fits the model on the data
        :param data:
         Tuple(ratings, users, items) where ratings is a dictionary {(user, item): rating},
         users are a dictionary {user: list of items the user has been exposed to},
         items are a dictionary {item: list of users that had been exposed to the item}
        :param kwargs:
        num_features - Number of latent variables
        num_iterations - Number of iterations to perform
        smoothness - Smoothness for random initialization
        shape_act - Shape of user activity
        rate_act - Rate of user activity
        shape_pop - Shape of item popularity
        rate_pop - Rate of item popularity
        shape_pref - Shape of user preferences
        shape_attr - Shape of item attributes
        :return:
        """
        super().fit(data, **kwargs)
        n_features = self.num_features
        num_iterations = int(kwargs.get('num_iterations', 10))
        smoothness = int(kwargs.get('smoothness', 100))
        user_activity_base_shape = float(kwargs.get('shape_act', 0.3))
        user_activity_base_rate = float(kwargs.get('rate_act', 1))
        item_popularity_base_shape = float(kwargs.get('shape_pop', 0.3))
        item_popularity_base_rate = float(kwargs.get('rate_pop', 1))
        user_preference_base_shape = float(kwargs.get('shape_pref', 0.3))
        item_attribute_base_shape = float(kwargs.get('shape_attr', 0.3))
        # Randomly initialize shapes and rates
        """
            kappa_shp   # User Activity Shape
            kappa_rte   # User Activity Rate
            gamma_shp   # User Preference Shape
            gamma_rte   # User Preference Rate
            tao_shp     # Item Popularity Shape
            tao_rte     # Item Popularity Rate
            lambda_shp  # Item Attribute Shape
            lambda_rte  # Item Attribute Rate
        """
        gamma_shp = {u: smoothness / np.random.gamma(smoothness, 1 / smoothness, n_features) for u in self.users}
        gamma_rte = {u: smoothness / np.random.gamma(smoothness, 1 / smoothness, n_features) for u in self.users}
        kappa_rte = {u: smoothness / np.random.gamma(smoothness, 1 / smoothness) for u in self.users}
        lambda_shp = {i: smoothness / np.random.gamma(smoothness, 1 / smoothness, n_features) for i in self.items}
        lambda_rte = {i: smoothness / np.random.gamma(smoothness, 1 / smoothness, n_features) for i in self.items}
        tao_rte = {i: smoothness / np.random.gamma(smoothness, 1 / smoothness) for i in self.items}
        # Compute initial user activity shape and item popularity shape
        kappa_shp = user_activity_base_shape + user_preference_base_shape * n_features
        tao_shp = item_popularity_base_shape + item_attribute_base_shape * n_features
        for _ in trange(num_iterations, desc='PMF'):
            # Update the multinomial
            phi = {(u, i): np.exp(
                special.digamma(gamma_shp[u]) - np.log(gamma_rte[u]) +
                special.digamma(lambda_shp[i]) - np.log(lambda_rte[i])
            ) for u in self.users for i in self.users[u]}
            phi = {ui: z / sum(z) for ui, z in phi.items()}
            for u in self.users:  # Update user preferences shape, rate and user activity rate
                gamma_shp[u] = user_preference_base_shape + sum(self.ratings[u, i] * phi[u, i] for i in self.users[u])
                gamma_rte[u] = (kappa_shp / kappa_rte[u]) + sum(lambda_shp[i] / lambda_rte[i] for i in self.users[u])
                kappa_rte[u] = (user_activity_base_shape / user_activity_base_rate) + sum(gamma_shp[u] / gamma_rte[u])
            for i in self.items:  # Update item attributes shape, rate and item popularity rate
                lambda_shp[i] = item_attribute_base_shape + sum(self.ratings[u, i] * phi[u, i] for u in self.items[i])
                lambda_rte[i] = (tao_shp / tao_rte[i]) + sum(gamma_shp[u] / gamma_rte[u] for u in self.items[i])
                tao_rte[i] = item_popularity_base_shape / item_popularity_base_rate
                tao_rte[i] += sum(lambda_shp[i] / lambda_rte[i])
        self.user_preferences_shape = gamma_shp
        self.user_preferences_rate = gamma_rte
        self.item_attributes_shape = lambda_shp
        self.item_attributes_rate = lambda_rte

    def predict(self, user, item):
        """
        Predicts the user rating for the item
        :param user: user id
        :param item: item id
        :return: Rating of the user for the item
        """
        rating = sum(
            np.exp(special.digamma(self.user_preferences_shape[user]) - np.log(self.user_preferences_rate[user]) +
                   special.digamma(self.item_attributes_shape[item]) - np.log(self.item_attributes_rate[item]))
        )
        # Fit to our rating range
        rating = max(rating, 0)
        rating = min(rating, 5)
        # rating = round(rating * 2) / 2
        return rating


def genre_plot(movies, title):
    """
    Plot genres histogram
    :param movies: Movies dataframe
    :param title: Graph title
    """
    genres = dict()
    for idx, row in movies.iterrows():
        for genre in row['genres'].split('|'):
            genres[genre] = genres.get(genre, 0) + 1
    if '(no genres listed)' in genres:
        genres.pop('(no genres listed)')
    genres = list(genres.items())
    genres.sort(key=lambda k: k[-1])
    genres, counts = list(zip(*genres))
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.title(title)
    plt.barh(genres, counts)
    plt.show()


def rating_plot(ratings, title):
    """
    Plot ratings histogram
    :param ratings: Ratings dataframe
    :param title: Graph title
    """
    rating_hist = dict()
    for idx, row in ratings.iterrows():
        rating = row['rating']
        rating_hist[rating] = rating_hist.get(rating, 0) + 1
    rating_hist = list(rating_hist.items())
    rating_hist.sort(key=lambda k: k[0])
    rating, counts = list(zip(*rating_hist))
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(np.arange(11) / 2)
    plt.title(title)
    plt.bar(rating, counts)
    plt.show()


def preprocess_meta(metadata):
    """
    Create a list of genres for every movie
    :param metadata: Genres dataframe
    :return: Dictionary {movie: list of genres for that movie}
    """
    genres = dict()
    with tqdm(total=len(metadata), desc='Loading metadata') as progress:
        for idx, row in metadata.iterrows():
            genres[int(row['movieId'])] = row['genres'].split('|')
            progress.update()
    return genres


def preprocess(data, include_users_movies=False):
    """
    Creates ratings: dictionary {(user, item): rating},
    users: dictionary {user: list of items the user has been exposed to} (Optional),
    items: dictionary {item: list of users that had been exposed to the item} (Optional)
    :param data:
    :param include_users_movies: Whether to include users and movies dictionaries
    :return: ratings [, users, items]
    """
    ratings = dict()
    users = dict()
    movies = dict()
    with tqdm(total=len(data), desc='Pre-processing Data') as progress:
        for idx, row in data.iterrows():
            u = int(row['userId'])
            m = int(row['movieId'])
            if include_users_movies:
                if u not in users:
                    users[u] = []
                if m not in movies:
                    movies[m] = []
                users[u].append(m)
                movies[m].append(u)
            ratings[u, m] = row['rating']
            progress.update()
    if include_users_movies:
        return ratings, users, movies
    return ratings


if __name__ == '__main__':
    # plt.style.use('dark_background')
    data_path = 'ml-1M'
    data = pd.read_csv(f'{data_path}/ratings.csv')
    metadata = pd.read_csv(f'{data_path}/movies.csv')
    # rating_plot(data, data_path)
    # genre_plot(metadata, data_path)
    train, test = train_test_split(data, test_size=0.2)
    # validation = train_test_split(train, test_size=0.25)
    train = preprocess(train, include_users_movies=True)
    # validation = preprocess(validation, include_users_movies=True)
    test = preprocess(test, include_users_movies=True)
    metadata = preprocess_meta(metadata)
    # for num_features in (20, 30, 40):
    #     for lambda_u in (1e-2, 1, 1e2, 1e3):
    #         for lambda_i in (1e-2, 1, 1e2, 1e3):
    #             for exposure in ('Bernoulli', 'Covariates'):
    #                 print(f'GMF {num_features}, {lambda_u}, {lambda_i}')
    #                 gmf = GaussianMF()
    #                 gmf.fit(
    #                     train,
    #                     genres=metadata,
    #                     num_features=num_features,
    #                     lambda_u=lambda_u,
    #                     lambda_i=lambda_i,
    #                     num_iterations=15,
    #                     exposure=exposure
    #                 )
    #                 # Choose by the best validation score
    #                 print('GMF Train Evaluation', gmf.evaluate(train))  # Evaluate train data
    #                 print('GMF Validation Evaluation', gmf.evaluate(validation))  # Evaluate train data
    # for num_features in (5, 10, 20):
    #     for shape in (1e-2, 1e-1, 1, 1e1):
    #         for rate in (1e-1, 1, 1e1):
    #             print(f'HPF {num_features}, {shape}, {rate}')
    #             pmf = PoissonMF()
    #             pmf.fit(
    #                 train,
    #                 num_features=num_features,
    #                 num_iterations=15,
    #                 smoothness=100,
    #                 shape_act=shape,
    #                 rate_act=rate,
    #                 shape_pop=shape,
    #                 rate_pop=rate,
    #                 shape_pref=shape,
    #                 shape_attr=shape
    #             )
    #             # Choose by the best validation score
    #             print('PMF Train Evaluation', pmf.evaluate(train))  # Evaluate train data
    #             print('PMF Validation Evaluation', pmf.evaluate(validation))  # Evaluate validation data
    gmf = GaussianMF()
    gmf.fit(
        train,
        genres=metadata,
        num_features=30,
        lambda_u=1e5,
        lambda_i=1e5,
        num_iterations=30,
        exposure='Bernoulli'
    )
    print('GMF Train Evaluation', gmf.evaluate(train))  # Evaluate train data
    print('GMF Test Evaluation', gmf.evaluate(test))  # Evaluate test data
    pmf = PoissonMF()
    pmf.fit(
        train,
        num_features=5,
        num_iterations=15,
        smoothness=100,
        shape_act=1,
        rate_act=0.3,
        shape_pop=1,
        rate_pop=0.3,
        shape_pref=1,
        shape_attr=1
    )
    print('PMF Train Evaluation', pmf.evaluate(train))  # Evaluate train data
    print('PMF Test Evaluation', pmf.evaluate(test))  # Evaluate test data
