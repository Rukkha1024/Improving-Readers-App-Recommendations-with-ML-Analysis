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


