import pandas as pd
import numpy as np

# 유저 데이터
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv("dataset/ml-100k/u.user", sep="|", names=u_cols, encoding="latin-1")
# print(users)

# 영화 데이터
# 2가지 이상의 장르에 1을 갖는 영화도 있음
# 원 핫 인코딩 형태임

i_cols = ['movie_id', 'title', 'release date', 'video release date',
'IMDB URL', 'unknown', 'Action', 'Adventure', 'Animation',
'Childerns\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
'Thriller', 'War', 'Western']
movies = pd.read_csv('dataset/ml-100k/u.item', sep="|", names=i_cols, encoding="latin-1")

# 유저 평점 데이터
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('dataset/ml-100k/u.data', sep='\t', names=r_cols, encoding="latin-1")

# row는 1차원, column은 2차원이므로.. axis는 0부터 ..
#print(ratings.drop('timestamp', axis=1))
ratings = ratings.drop('timestamp', axis=1)

# 인덱스 설정안하고, 무비 id랑 title만 추출(다른 데이터 제거)
movies = movies[['movie_id', 'title']]

# x, 데이터 원본 보존, y, user_id를 기준으로 나누기 위함
x = ratings.copy()
y = ratings['user_id'] # stratified sampling 방식

# 훈련 / 테스트 데이터 25% 로 분리
split_index = int(len(x)*0.75)
x_train = x[:split_index]
x_test = x[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
# print(y_train, y_test)


# Objective Function
# RMSE 정확도 계산
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# 모델별 RMSE 계산 함수(해당 모델의 결과값과 실제 값의 RMSE값 도출  )
def score(model):
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    y_true = np.array(x_test['rating'])
    
    return RMSE(y_true, y_pred)

# train 데이터로 Full Matrix 구하기
# 유저id를 인덱스로, 유저가 영화에 부여한 평점 매트릭스로 피버팅함
rating_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')
# print(rating_matrix)


# 실제 모델, 전체 평균으로 예측치를 계산하는 기본 모델 (예측 모델)
def best_seller(user_id, movie_id):
    # train set에는 존재하지 않지만 test set에 존재하는 영화로 인해 발생하는 오류 방지 (try-except)
    try:
        rating = train_mean[movie_id]
    except: 
        rating = 3.0
    return rating

# 영화의 평점 평균 집계
train_mean = x_train.groupby(['movie_id'])['rating'].mean()

# 모델 실행, 결과적으로 RMSE값이 증가함. 자신의 테스트 값으로 test하지 않았으므로 오차율이 증가한 것임
# print(score(best_seller))
