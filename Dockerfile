# 1
FROM python:3.12

# 2
RUN pip install "fastapi[standard]" gensim nltk scikit-learn

# 3
COPY ./ /app
WORKDIR /app

# 4
CMD exec fastapi run