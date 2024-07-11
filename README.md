*This project was conducted as part of the 'Data Prediction Model' class during the 2023-1*

# Introduction 

## What is '리더스' 

<img src="https://github.com/Rukkha1024/Improving-Readers-App-Recommendations-with-ML-Analysis/assets/128712657/c7b71d73-baf5-4b3d-9474-6b770356b249" width="10%" height="10%"></img>

'리더스' 어플은 사용자가 읽은 책을 기록하고, 마음에 드는 구절을 강조하며 감상평을 남길 수 있는 독서 플랫폼이다. 기록을 통해 사용자는 자신의 독서 이력을 관리 및 공유하며, '북클럽'이라는 기능을 활용해 '리더스'가 선정한 다양한 주제의 책을 다른 어플 사용자들과 함께 읽는다. 


## Purpose of the project 

1. 사용자의 책 스크랩 문구를 분석하는 텍스트 마이닝 

2. 개인화된 책을 추천하는 알고리즘 개발 

추천 알고리즘을 통해 북클럽에서 사용자의 취향과 기준에 따른 최적의 책을 추천받으며, 스크랩 문구 텍스트 마이닝을 연령에 따라 분류하여 가장 높은 빈도수를 가진 단어를 각 연령 그룹의 대표 키워드로 설정해 개인화된 독서경험을 제공한다. 

---

# 1. tf-idf 기반 도서 및 문구 추천 알고리즘 

## Purpose 

이용자가 자신이 읽은 책에서 인상 깊었던 부분에 대한 스크랩 문구(구절)이 유저들 간 교류를 위한 합리적인 수단이라 판단했으며, 비슷한 특성의 유저 집단을 대표할 수 있는 키워드를 선정 후 키워드 관련 도서와 문구를 추천해준다면 유전 간 교유를 활발히 하고 '리더스'만의 독서 SNS라는 강점을 확고히 할 것이다. 




## Processing(Code)

tf-idf 알고리즘을 기반으로 유저들을 연령대별로 그룹화 한 후 tf-idf가 높은 단어(키워드)를 각 그룹을 대표하는 키워드로 선정한다. 해당 키워드를 통해 그룹 내 유저들에게 '추천도서', '추천문구', '페이지'를 추천해주는 서비스를 제안한다. 

<details> 

<summary> 1. 전처리 </summary>

``` R
library(readxl)
library(dplyr)
library(stringr)
library(KoNLP)
library(tidyr)
library(tidytext)
library(ggplot2)

# 데이터셋 불러오기
scrap <- read_excel("08_scrap.xlsx")
user <- read_excel("01_user.xlsx")
user_book <- read_excel("04_user_book.xlsx")
book <- read_excel("05_book.xlsx")


# 중복된 book_id와 id를 title로 변경
scrap <- scrap %>%
  inner_join(book %>% distinct(id, title), by = c("book_id" = "id")) %>%
  select(-book_id) %>%
  rename(book_title = title)

# book_id를 기준으로 도서의 평점이 4 이상인 도서만 필터링
filtered_books <- user_book %>% dplyr::filter(rate >= 4)

# 중복 id를 기준으로 유저의 나이, content를 필터링하여 age_content 변수에 저장
age_content <- scrap %>% 
  inner_join(user, by = "user_id") %>% 
  select(age = birth_year, content, book_title, page) %>% 
  mutate(age = as.numeric(format(Sys.Date(), "%Y")) - as.numeric(age) + 1)

# 결측치 제거 및 20세 미만 값 제거
age_content <- na.omit(age_content)
age_content <- age_content %>% dplyr::filter(age > 20)

# 전체 데이터의 40%만 랜덤 추출
set.seed(123)
sample_age_content <- age_content %>% sample_frac(0.4)
```
</details>

<details> 

<summary> 2. 텍스트 정제 </summary>

``` R 
# 한글 아닌 글자 및 모든 특수문자와 이모티콘 공백으로 변경
sample_age_content$content <- str_replace_all(sample_age_content$content, 
                                            "[^[:alpha:]ㄱ-ㅎㅏ-ㅣ가-힣]", " ")
sample_age_content$content <- str_squish(sample_age_content$content)

# 영어와 러시아어를 제거하는 함수 정의
remove_english_and_russian <- function(text) {
  str_replace_all(text, "[a-zA-Zа-яА-Я]", "")
}

# 'ㅡ'와 단일한 자음과 모음 및 한자를 제거하는 함수 정의
remove_dash <- function(text) {
  pattern <- "[-ㅋㅎㅠㅡㅜ\u4E00-\u9FFF]"
  str_replace_all(text, pattern, "")
}

# 조사 제거하는 함수 정의
remove_particles <- function(text) {
  str_replace_all(text, "\\s[은는이가을를과와\\s]", " ")
}

# "content" 열에서 불필요한 문자 제거
sample_age_content$content <- sapply(sample_age_content$content, 
                                     remove_english_and_russian)
sample_age_content$content <- sapply(sample_age_content$content, remove_dash)
sample_age_content$content <- sapply(sample_age_content$content, remove_particles)
sample_age_content$content <- str_squish(sample_age_content$content)

``` 

</details> 

<details> 

<summary>  3. 토큰화 및 빈도수 측정  </summary>

``` R 
# 명사 추출 후 토큰화 
noun_tokenizer <- function(x) {
  words <- unlist(extractNoun(x))
  words <- setdiff(words, c("교보", "교보에서","을","를",
                            "은","는","이","가")) 
  paste(words, collapse = " ")
}

# "content"에 대해서 명사를 추출하고 "nouns" 열을 생성
sample_age_content <- sample_age_content %>%
  mutate(nouns = sapply(content, noun_tokenizer))

# "nouns" 열의 내용을 기준으로 토큰화
output <- sample_age_content %>%
  unnest_tokens(word, nouns)

# 결과 확인
print(output$word)

# 토큰화된 명사들의 빈도수
word_frequency <- output %>%
  count(word, sort = TRUE) %>% # 단어 빈도 구해 내림차순 정렬
  dplyr::filter(nchar(word) > 1) # 두 글자 이상만 남기기

# 결과 확인
print(word_frequency)


``` 

</details> 

<details> 

<summary>  4. 연령대별 tf-idf 계산 및 결과   </summary>

``` R 
# 연대 변수 추가
output <- output %>%
  mutate(age_group = case_when(
    age >= 20 & age < 30 ~ "20대",
    age >= 30 & age < 40 ~ "30대",
    age >= 40 & age < 50 ~ "40대",
    age >= 50 ~ "50대 이상",
  ))

# 단어 빈도수 연령별로 정렬
word_frequency <- output %>%
  count(age_group, word, sort = TRUE) %>%
  dplyr::filter(nchar(word) > 1)

# 연령별 Tf-idf 계산
word_tfidf <- word_frequency %>%
  bind_tf_idf(word, age_group, n)

# Tf-idf를 기준으로 내림차순 정렬
sorted_word_tfidf <- word_tfidf %>%
  arrange(desc(tf_idf))

# 연령대별 Tf-idf 확인
sorted_word_tfidf[sorted_word_tfidf$age_group=='20대',]
sorted_word_tfidf[sorted_word_tfidf$age_group=='30대',]
sorted_word_tfidf[sorted_word_tfidf$age_group=='40대',]
sorted_word_tfidf[sorted_word_tfidf$age_group=='50대 이상',]


``` 

</details> 


<details> 

<summary>  5. 시각화   </summary>

``` R 
# 시각화
ggplot(top_words, aes(x = reorder_within(word, tf_idf, age_group),
                      y = tf_idf,
                      fill = age_group)) +
  geom_col(show.legend = F) +
  coord_flip() +
  facet_wrap(~ age_group, scales = "free", ncol = 2) +
  scale_x_reordered() +
  labs(x = NULL)

``` 

</details> 

<details> 

<summary>  6. 도서 및 문구 추천   </summary>

``` R 
# 각 연령대별로 가장 높은 Tff 값을 가진 단어 선택
top_tfidf_words <- sorted_word_tfidf %>%
  group_by(age_group) %>%
  top_n(1, wt = tf_idf) %>%
  ungroup() %>%
  select(age_group, word) %>%
  arrange(age_group)

# 각 연령대에서 상위 단어를 포함하는 content, book_title, page 선택하는 함수
get_top_word_examples <- function(age_group, top_word, df, n = 3) {
  results <- df %>%
    dplyr::filter(age_group == age_group & grepl(top_word, content)) %>%
    sample_n(min(n, nrow(.))) %>%
    select(book_title, content, page)
  colnames(results) <- c("추천 도서", "추천 스크랩 문구", "페이지")
  return(results)
}

# 각 연령대별 결과 추출
results_20s <- get_top_word_examples("20대", 
                                     top_tfidf_words$word[top_tfidf_words$age_group == "20대"], 
                                     sample_age_content)

results_30s <- get_top_word_examples("30대", 
                                     top_tfidf_words$word[top_tfidf_words$age_group == "30대"], 
                                     sample_age_content)

results_40s <- get_top_word_examples("40대", 
                                     top_tfidf_words$word[top_tfidf_words$age_group == "40대"], 
                                     sample_age_content)

results_50s <- get_top_word_examples("50대 이상", 
                                     top_tfidf_words$word[top_tfidf_words$age_group == "50대 이상"], 
                                     sample_age_content)

# 연령별 top 키워드 추천 문구, 도서, 페이지 추천 (3쌍씩)
print(results_20s)
print(results_30s)
print(results_40s)
print(results_50s)

``` 

</details> 


## Results 

20대의 경우, 추천 도서가 모두 자기계발 도서임을 통해 다른 그룹에 비해 자기계발에 관심을 두는 경향이 있다고 볼 수 있으며, 대표 키워드를 '미라클' 또는 '자기계발'로 뽑을 수 있다. 30대의 경우, 추천 도서 모두가 박노해 시인의 시집인 것을 통해 문학에 관심을 두는 경향이 있고 대표 키워드를 '음유시인' 또는 '박노해'라고 할 수 있다. 40대는 추천 도서가 모두 신지식론이며, 대표 키워드를 '크리스찬' 혹은 '독실한 신자'가 될 것이다. 마지막으로 50대 이상의 연령층에서는 추천 스크랩 문구에 공통적으로 '나이들수록'이라는 단어가 보여지고 따라서 '위로', '성찰'과 같은 단어들을 키워드로 제안할 수 있을 것이다. 


## Limitation 

- 도서 내 추출된 스크랩 문구와 유저가 자체적으로 생성한 메모 형식의 스크랩 데이터 분리의 필요성 
- 자연어 처리 과정 중 한글 텍스트 정제의 한계 

---

# 2. 책 카테고리 별 북클럽 책 추천 서비스 

## Purpose 

북클럽 서비스는 독서 경험이 부족한 독자들에게 적합한 책을 고를 수 있도록 카테고리 별로 선정된 다양한 책들을 소개하며, 읽고 싶은 책을 찾게 된 독자들이 공통의 책을 찾은 모임에 가입 할 수 있게 한다. 하지만 2023년 6월 기준 북클럽 서비스는 2022년 3월 이후로 잠정 연기되어 있다. 이를 위해 북클럽 서비스를 발전시키고자, 머신러닝을 포함한 추천 알고리즘을 도입하고자 한다. 



## Processing(Code)


<details> 


<summary> 1. 전처리 및 알고리즘 변수 만들기  </summary>


``` R
library(readr)
library(readxl)
library(VIM)
library(caret)
library(glmnet)

# Data Import ##################################################################
user <- read_excel("01_user.xlsx")
user_cat <- read_excel("02_user_cat.xlsx")
follow <- read_excel("03_follow.xlsx")
user_book <- read_excel("04_user_book.xlsx")
book <- read_excel("05_book.xlsx")
book_cat <- read_excel("06_book_cat.xlsx")
cat <- read_excel("07_cat.xlsx")
scrap <- read_excel("08_scrap.xlsx")
book_book.club <- read_excel("book_bookclub.xlsx")

# book_cat과 cat을 book_category_id를 기준으로 합친다 ------
A <- inner_join(book_cat,cat,by='book_category_id')
A

# book_id 별로 그룹해서 모든 카테고리를 하나의 행으로 만든다:cat_one.row  -----
# Merge the data in 'name', 'depth_1' to 'depth_5' columns
result <- A %>% select(-book_category_id) %>%
  group_by(book_id) %>% 
  mutate(final_columns = paste(name, depth_1, depth_2, depth_3, depth_4, depth_5, sep = "/")) %>%
  select(book_id, final_columns)

merged_result <- result %>%
  group_by(book_id) %>%
  summarise(final_columns = paste(final_columns, collapse = "/"))

cat_one.row <- merged_result %>%
  mutate(final_columns = gsub("/NA", "", final_columns)) %>% 
  mutate(final_columns = gsub(">", "", final_columns)) %>% 
  mutate(final_columns = gsub("/", " ", final_columns))

# cat_one.row + book: merge_2 ------
# 영어로 된 책 제거
book <- subset(book, !grepl("[a-zA-Z]", title))
names(book)[1] = ('book_id')

# 출판사 별로 책 개수 세기
publisher_num <- book %>% group_by(publisher) %>% 
  mutate(publisher_num = n_distinct(book_id)) %>% ungroup()

merge_2 <- inner_join(publisher_num,cat_one.row,by='book_id')
merge_2 <- merge_2 %>% select(book_id, page, final_columns, publisher_num)


# user_book + cat_one.row + book: merge_3 ####
# 각 도서별로 status 비율을 구해서 합침 
# Count read_status occurrences within each book_id
status_counts <- user_book %>%
  group_by(book_id, read_status) %>%
  summarise(count = n()) %>%
  ungroup()

# Calculate ratios of read_status by book_id
status_ratios <- status_counts %>%
  group_by(book_id) %>%
  mutate(ratio = count / sum(count)) %>%
  ungroup()

status_ratios_wide <- 
  status_ratios %>% 
  spread(key = read_status, value=ratio) %>% 
  replace_na(list(READ_STATUS_BEFORE = 0, READ_STATUS_ING = 0, READ_STATUS_DONE = 0, READ_STATUS_PAUSE = 0)) %>%
  select(book_id, READ_STATUS_BEFORE, READ_STATUS_ING, READ_STATUS_DONE, READ_STATUS_PAUSE) %>% 
  group_by(book_id) %>% 
  summarise(READ_STATUS_BEFORE = max(READ_STATUS_BEFORE),
            READ_STATUS_ING = max(READ_STATUS_ING),
            READ_STATUS_DONE = max(READ_STATUS_DONE),
            READ_STATUS_PAUSE = max(READ_STATUS_PAUSE))



# 전체 책들의 평균 평점, 평점 준 인원, 책장에 담은 인원
book_all <- inner_join(book, user_book, by='book_id', multiple='all')

# 책장에 담은 인원의 수

count_user_lab <- book_all %>% 
  group_by(book_id) %>% 
  summarise(count_user = n_distinct(user_id))

# 평점 평균, 평균 준 사람의 수
rate_lab <- book_all %>%
  dplyr::filter(!is.na(rate), rate > 0) %>%
  group_by(book_id) %>%
  summarise(avg_rate = mean(rate), 
            book_rating_user = n_distinct(user_id))

# count_user_lab + rate_lab: book_all_merged
book_all_merged <- left_join(count_user_lab, rate_lab, by='book_id')

# status_ratios_wide + book_all_merged
mereged <- inner_join(status_ratios_wide, book_all_merged, by='book_id')

# merge
merge_3 <- inner_join(merge_2, mereged, by='book_id')

# 책 별 스크랩 수
scrap_num <- scrap %>% group_by(book_id) %>% 
  summarise(scrap_num = n_distinct(content))

merge_4 <- full_join(merge_3, scrap_num, by = 'book_id')
merge_4$scrap_num[is.na(merge_4$scrap_num)] <- 0

```
</details>


<details> 

<summary> 2. 책 카테고리 클러스터링 </summary>

```R
library(stringr)
library(textclean)
library(tidytext)
library(KoNLP)
library(tm)
library(topicmodels)
library(writexl)
library(ldatuning)  

raw_genre <- read.csv('cat_one.row.csv')
str(raw_genre)


genre <- raw_genre %>% 
  select(final_columns) %>% 
  mutate(final_columns = str_replace_all(final_columns, "[^가-힣]", " "),
         final_columns = str_squish(final_columns),
         id = raw_genre$book_id)
glimpse(genre)
load("genre.RData")

#----------------------------------------------------------------
# 토큰화 하기
# 1. 형태소 분석기를 이용해 품사 기준으로 토큰화하기
library(tidytext)
library(KoNLP)
load("genre_pos.RData")
genre_pos <- genre %>% 
  unnest_tokens(input = final_columns,
                output = word,
                token = extractNoun,
                drop = F) %>% 
  dplyr::filter(str_count(word)>1)
head(genre_pos)

# 빈도 높은 단어 제거
# 빈도 수 설정을 위해 함수 생성
calculate_dim <- function(n_value) {
  lab <- genre_pos %>% 
    add_count(word) %>% 
    dplyr::filter(n <= n_value) %>% 
    select(-n)
  
  lab_count_word <- lab %>% 
    add_count(word)
  
  lab_count_word_doc <- lab_count_word %>% count(id, word, sort=T)
  
  lab_dtm_comment <- lab_count_word_doc %>%
    cast_dtm(document = id, term = word, value = n)
  
  lab_lda_model <- LDA(lab_dtm_comment,
                       k = 2,
                       method = "Gibbs",
                       control = list(seed = 223))
  
  lab_doc_topic <- tidy(lab_lda_model, matrix = "gamma")
  
  lab_doc_class <- lab_doc_topic %>%
    group_by(document) %>%
    slice_max(gamma, n = 1)
  
  lab_doc_class <- lab_doc_class %>% 
    ungroup() %>%
    mutate(lab_doc_class = as.integer(document))
  
  return(dim(lab_doc_class))
}

calculate_dim(10000)

n_values <- seq(1000, 20000, by = 1000)

dim_values <- sapply(n_values, calculate_dim)
dim_1000.20000 <- dim_values
# 5000으로 설정 시 book_id 개수가 89348개로 줄어듬. 5000개를 기준으로 해서 필터링

lab <- genre_pos %>% 
  add_count(word) %>% 
  dplyr::filter(n <= 9000) %>% 
  select(-n)
str(lab)
lab %>% select(id, word) %>% head

#-------------------------------------------------------------------------------
# 단어 빈도 세기
library(tm)
lab_count_word <- lab %>% 
  add_count(word)
str(lab_count_word)

# 문서별 단어 빈도 구하기
lab_count_word_doc <- lab_count_word %>% count(id, word, sort=T)
head(lab_count_word_doc)

#-------------------------------------------------------------------------------
# DTM 만들기
lab_dtm_comment_5000 <- lab_count_word_doc %>%
  cast_dtm(document = id, term = word, value = n)
lab_dtm_comment_5000
# save(lab_dtm_comment_5000, file = 'lab_dtm_comment.RData')
# load('lab_dtm_comment.RData')

# LDA 모델: number of topics - k
library(topicmodels)
lab_lda_model <- LDA(lab_dtm_comment_5000,
                 k = 130,
                 method = "Gibbs",
                 control = list(seed = 223))
# save(lab_lda_model, file = 'lab_lda_model.RData')
# load('lab_lda_model.RData')
glimpse(lab_lda_model)

#-------------------------------------------------------------------------------
# 최적의 토픽 수 도출
# 하이퍼 파라미터 튜닝으로 토픽 수 정하기
# 1. 토픽 수 바꿔가며 LDA 모델 여러개 만들기
library(ldatuning)  
# load('lab_dtm_comment.RData')
lab_models <- FindTopicsNumber(dtm = lab_dtm_comment_5000, 
                               topics = 50:180,
                               return_models = T,
                               control = list(seed = 1234))
# save(lab_models,file = 'lab_models.RData')
load('lab_models.RData')

lab_models_for_appropriate_topics <- lab_models %>%
  select(topics, Griffiths2004)
lab_models_for_appropriate_topics

# 2. 최적의 토픽 수 정하기
FindTopicsNumber_plot(lab_models)

#-------------------------------------------------------------------------------
# 토픽별 단어 확률, beta 추출하기
lab_term_topic <- tidy(lab_lda_model, matrix = 'beta')

# 토픽별 book_id 빈도 구하기
lab_count_topic <- lab_doc_class %>% count(topic)
write.csv(lab_count_topic, file = 'lab_count_topic.csv',fileEncoding = 'UTF-8')


# 모든 토픽의 주요 단어 살펴보기
terms(lab_lda_model, 20) %>%
  data.frame()                  -> lab_topic_20words
write_xlsx(lab_topic_20words, 'lab_topic_20words.xlsx')

#-------------------------------------------------------------------------------
# 문서별 확률이 가장 높은 토픽으로 분류하기
# 문서별 토픽 확률 gamma 추출하기
lab_doc_topic <- tidy(lab_lda_model, matrix = "gamma")
lab_doc_topic
# 문서별로 확률이 가장 높은 토픽 추출하기
lab_doc_class <- lab_doc_topic %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>% 
  ungroup %>% 
  mutate(lab_doc_class = as.integer(document))
lab_doc_class
```

</details> 


<details> 

<summary> 3. 유저 관심사 라벨링 </summary>

``` R 
# user_cat --------------------------------------------------------------------
str(user_cat)
glimpse(user_cat)
user_cat %>% distinct(title) %>% data.frame()

# Label encoding
user_cat$label <- factor(user_cat$title, 
                         levels = unique(user_cat$title), 
                         labels = 1:length(unique(user_cat$title)))

# Update the label column
# 습관 -> 종교
# 시 = 문학 / 다이어트 = 운동|레저 / 인공지능 = 데이터
# 목표달성 = 자기계발 / 영어 = 언어
# 기타: 원서, 만화, 라노벨 등
user_cat_label <- 
  user_cat %>%
  mutate(label = recode(
    label,
    "4" = "15",
    "5" = "22",
    "11" = "19",
    "13" = "12",
    "20" = "1"
  )) %>% as_tibble()

user_cat_label %>% select(title, label) %>% unique() %>% data.frame()

# topic clustering -------------------------------------------------------------
# topic 1과 93은 분류상 어려움으로 제거하였음
lab_topic <- read_csv("lab_topic.csv")
lab_topic <- lab_topic %>% select(topic, user_cate) %>% na.omit() %>% as.data.frame()

str(lab_doc_class)
glimpse(lab_doc_class)
lab_doc_class %>% distinct(topic) %>% as.data.frame()
topic_id <- lab_doc_class %>% select(topic, lab_doc_class) %>% as.data.frame()

# lab_topic, topic_id 병합
topic_label_id <- merge(lab_topic, topic_id, by='topic')
colnames(topic_label_id)[colnames(topic_label_id) == "user_cate"] <- "label"
colnames(topic_label_id)[colnames(topic_label_id) == "lab_doc_class"] <- "book_id"
head(topic_label_id)

# user_id랑 병합: b
user_cat_label$label <- as.numeric(user_cat_label$label)
user_cat_label <- user_cat_label[,-2]
b <- left_join(a, user_cat_label, by='label', multiple = 'all')
b <- b[,-2]
head(b)

# topic_label_id랑 모델 원본 데이터랑 결합(books_for_model)
label_books <- left_join(books_for_model, topic_label_id, by='book_id', multiple = 'all')


``` 

</details> 


<details> 

<summary> 4. 알고리즘 모델 </summary>

``` R 
# 카테고리 정보 제거 -----------------------------------------------------------
library(VIM)
readers_book <- merge_4[,-3]
readers_book <- na.omit(readers_book)
VIM::aggr(readers_book)

str(readers_book)
glimpse(readers_book)

# book_club 책들만 따로 떼어내기 -----------------------------------------------
# test_data = book_club_var
colnames(book_book.club)[1] <- 'book_id'
book_club <- book_book.club[,1]


book_club_var <- inner_join(book_club, readers_book,
                           by='book_id')
VIM::aggr(book_club_var)

# 교보문고 책들만 따로 떼어내기 ------------------------------------------------
# train data = kyobo_data
library(readr)
Kyobo_ALL <- read_csv("Kyobo_ALL.csv")
str(Kyobo_ALL)
library(readxl)
book <- read_excel("05_book.xlsx")
na.omit(book)
book$isbn13 <- as.numeric(book$isbn13)
kyobo <- inner_join(Kyobo_ALL, book, by = 'isbn13')
kyobo_unique <- kyobo[!duplicated(kyobo$isbn13), ]
kyobo_unique %>% view()

kyobo_data <- kyobo_unique[,3]
colnames(kyobo_data) <- 'book_id'

kyobo_var <- inner_join(kyobo_data, readers_book,
                           by='book_id')
VIM::aggr(kyobo_var)
kyobo_var

# label a dependent variable: 1,0 ----------------------------------------------------------
# 북클럽&교보문고 == 1, the others == 0
standard <- rbind(book_club_var, kyobo_var)
standard <- standard %>% mutate(var = 1)
non_standard <- readers_book %>% mutate(var = 0)

books_for_model <- rbind(standard, non_standard)
books_for_model$var <- as.factor(books_for_model$var)
# save(books_for_model, file = 'books_for_model.RData')
# load('books_for_model.RData')

# logistic linear regression model ------------------------------------------------------------
# model
library(caret)
library(glmnet)
set.seed(123)
splitIndex <- createDataPartition(books_for_model$var, p = 0.8, list=F)
train_set <- books_for_model[splitIndex, ]
test_set <- books_for_model[-splitIndex, ]

glm_model <- glm(formula = var~ page + READ_STATUS_BEFORE +
                   READ_STATUS_ING + READ_STATUS_DONE + READ_STATUS_PAUSE +
                   count_user + avg_rate + book_rating_user + 
                   scrap_num + publisher_num,
                 family = binomial, data = train_set)
summary(glm_model)
glm_model$coefficients

# Predict the model ------------------------------------------------------------
# predict
glm_pred <- predict(glm_model, newdata = test_set, type = 'response')

# Coerce glm_pred to a data frame
glm_pred_df <- as.data.frame(glm_pred)

# Add a new column for book_id from the test_set
glm_pred_df$book_id <- test_set$book_id

# Sort the data frame to see the top 10 results
top_results <- glm_pred_df %>% arrange(desc(glm_pred_df)) %>% head(10)
print(top_results)
book_id_title <- book %>% rename('book_id' = 'id') %>% select(book_id, title)
top10_results <- left_join(top_results,book_id_title, by = 'book_id', multiple='all')
top10_results

# Calculate the model-----------------------------------------------------------
# Load necessary libraries
library(pROC)

# Create ROC curve
roc_obj <- roc(test_set$var, glm_pred)
roc_obj
# Plot ROC curve
plot_obj <- plot(roc_obj, print.auc=TRUE, print.auc.cex=7)

# Predict class based on the threshold
glm_pred_class <- ifelse(glm_pred > 0.5, 1, 0)

# Convert to factor
glm_pred_class_factor <- as.factor(glm_pred_class)
test_var_factor <- as.factor(test_set$var)

# Create confusion matrix
cm <- confusionMatrix(glm_pred_class_factor, test_var_factor)
save(cm, file = 'cm.RData')
# Print the confusion matrix
print(cm$table)

# Precision, Recall, Sensitivity, Specificity, Accuracy can be extracted from the confusionMatrix object
precision <- cm$byClass['Pos Pred Value']
recall <- cm$byClass['Sensitivity']
specificity <- cm$byClass['Specificity']
accuracy <- cm$overall['Accuracy']

# Print metrics
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("Sensitivity: ", recall, "\n") # recall and sensitivity are the same
cat("Specificity: ", specificity, "\n")
cat("Accuracy: ", accuracy, "\n")


``` 

</details> 

<details> 

<summary> 5. 유저의 관심사로 필터링한 책 추천 </summary>

``` R 
test_book <- 
  label_books %>% 
  dplyr::filter(label == 8) %>% 
  distinct(.,book_id,.keep_all=T) %>% 
  select(-topic, -label)

test_book_pred <- predict(glm_model, newdata = test_book, type = 'response')

# Coerce glm_pred to a data frame
test_book_pred <- as.data.frame(test_book_pred)

# Add a new column for book_id from the test_set
test_book_pred$book_id <- test_book$book_id

# Sort the data frame to see the top 10 results
top_results <- test_book_pred %>% arrange(desc(test_book_pred)) %>% head(10)

# top10 result와 책 제목 결합
book_id_title <- book %>% rename('book_id' = 'id') %>% select(book_id, title)
top10_results <- left_join(top_results,book_id_title, by = 'book_id', multiple='all')
top10_results



``` 

</details> 





## Results 

| book_id | title                                                             |
| ------- | ----------------------------------------------------------------- |
| 626     | 돈, 뜨겁게 사랑하고 차갑게 다루어라                                              |
| 3382    | 왜 주식인가 - 부자가 되려면 자본이 일하게 하라                                       |
| 4337    | 워런 버핏과의 점심식사 - 가치투자자로 거듭나다                                        |
| 402481  | 현명한 투자자 - 벤저민 그레이엄 직접 쓴 마지막 개정판, 개정4판                             |
| 135689  | 네이버 증권으로 배우는 주식투자 실전 가이드북 - 주식 고수들만 아는 '네이버 증권 200% 활용법!', 개정증보판  |
| 625     | 전설로 떠나는 월가의 영웅 - 13년간 주식으로 단 한 해도 손실을 본 적이 없는 피터린치 투자, 2017 최신개정판 |
| 17143   | 투자에 대한 생각 - 월스트리트가 가장 신뢰한 하워드 막스의 20가지 투자 철학                      |
| 506     | 위대한 기업에 투자하라                                                      |
| 1721576 | 강방천 & 존리와 함께하는 나의 첫 주식 교과서 - 기본부터 제대로 배우는 평생 투자 원칙                |
| 121002  | 기업공시 완전정복 - 경영 전략과 투자의 항방이 한눈에 보이는                                |

로지스텍 회귀 모델 성능에 영향을 미치는 변수는 완독률과 평균평점이며, 리더스 책 카테고리 별 분류 데이터를 'KoNLP' 패키지를 통해 유저의 관심분야에 맞게 재분류하여 책을 독자에게 추천할 수 있게 한다. 위 표는 재테크와 관련된 상위 10개 항목을 추천한 리스트이다. 



## Limitation 

- 머신러닝 알고리즘의 낮은 specificity 
- 자연어 처리 과정 중 한글 텍스트 정제의 한계 
- 특정 분야의 책들에 편중된 예측값 
- 기존 리더스 어플 내 이용자 관심 분야 세분화 부족 
