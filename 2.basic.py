import pandas as pd
import numpy as np

# 유저 데이터
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zipcode']
users = pd.read_csv('dataset/ml-100k/u.user', sep="|", names=u_cols)
users = users.set_index('user_id')
print(users.head())

# 영화 데이터
i_cols = ['movie_id', 'title', 'release date', 'video release date',
'IMDB URL', 'unknown', 'Action', 'Adventure', 'Animation',
'Childerns\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
'Thriller', 'War', 'Western']

# 장르는 boolean으로 나타냄, Sparse 함
# 2가지 이상의 장르에 1을 갖는 영화도 있음
movies = pd.read_csv('dataset/ml-100k/u.item', sep="|", names=i_cols, encoding="latin-1")
movies = movies.set_index('movie_id')
print(movies.head())


# 유저 평점 데이터
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('dataset/ml-100k/u.data', sep='\t', names=r_cols,
encoding='latin-1')
ratings = ratings.set_index('user_id')
print(ratings.head())

# 영화 평점은 사용자 - 영화 - 평점으로 이루어져있음

# 인기 제품 방식(Best-Seller 추천)
def recom_movie1(n_items):
    # 평점 평균 내림차순 정렬
    # 상위부터 지정된 아이템까지 저장
    movie_sort = movie_mean.sort_values(by='rating', ascending=False)[:n_items]
    recom_movies = movies.iloc[movie_sort.index] # 상위 영화 조회
    recommendations = recom_movies['title'] # 상위 영화의 타이틀을 리턴
    return recommendations

# 한줄로 줄일 수 있는 코드도 존재
# P 17. recom_movie2

# 영화 별로 묶고, 평점 평균을 구함
movie_mean = ratings.groupby(['movie_id']).mean()
print(movie_mean)
print(recom_movie1(5))

# 정확도 측정
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# best-seller 방식의 RMSE 계산
rmse = []
# 평점데이터에서 유저를 받아옴 (index가 유저 id)
for user in set(ratings.index):
    # 실제 y 데이터는 유저가 부여한 평점
    y_true = ratings.loc[user]['rating']
    # 평점 데이터의 유저 값 select 후 영화 데이터 불러옴
    # 평점 평균이 해당 영화의 예측 값임 (best-seller)
    # iloc은 행번호, loc은 label 명이라고 보면 된다. loc이 앱도적으로 많이 사용됨
    y_pred = movie_mean.loc[ratings.loc[user]['movie_id']]['rating']
    # print(y_pred)
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)

# 집단 추천방법은 0.996만큼 오차가 발생하는 것을 볼 수 있다.
print(np.mean(rmse))