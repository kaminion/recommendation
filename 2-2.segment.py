import pandas as pd

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
print(y_train, y_test)