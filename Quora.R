library(data.table)
library(Metrics)
library(xgboost)
library(stringi)
library(tm)
library(text2vec)
library(wordnet)
library(tokenizers)
library(xgboost)
library(pROC)

MultiLogLoss <- function(act, pred){
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

getWordWeights <- function(count,eps=5000,min_count=2) {
  if (count < min_count) {0} else
    {1 / (count + eps)}
}

setwd <- "C:/R-Studio/Quora"
train  <- fread( "./train.csv")
test  <- fread( "./test.csv")

train$is_duplicate <- as.numeric(train$is_duplicate)

# train$question1CountWords <- sapply(train$question1,stri_count_words,USE.NAMES=FALSE)
# train$question2CountWords <- sapply(train$question2,stri_count_words,USE.NAMES=FALSE)

# Create a corpus
dfCorpus = Corpus(VectorSource(c(train$question1,train$question2)))
inspect(dfCorpus)
myTdm <- TermDocumentMatrix(dfCorpus, control=list(tolower=TRUE))

word.count <- slam::row_sums(myTdm)
word.weight <- sapply(word.count, function(x) getWordWeights(x,5000,2))
  
# seldomWords <- names(word.count[word.count < 15000])
# frequentWords <- names(word.count[word.count >= 15000])
stopWords <- tm::stopwords(kind="en")

#dict <- unlist(strsplit(paste(tolower(train$question1), tolower(train$question2))," "))
# dict <- table(unlist(strsplit(paste(tolower(train$question1), tolower(train$question2))," ")))
# word.weight <- sapply(dict, function(x) getWordWeights(x,5000,2))

nGramHitRate <- function(question1,question2,n=1,n_min=1) {
  t1 <- unlist(tokenize_ngrams(question1,n=n, n_min=n_min))
  t2 <- unlist(tokenize_ngrams(question2,n=n, n_min=n_min))
  sharedPhrases <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  hitRate <- 2*length(sharedPhrases)/bothLengths
  hitRate <- ifelse(is.infinite(hitRate),0,hitRate)
}

sampleSize <- 20000
sample <- sample(1:nrow(train),sampleSize)
print(length(sample))

j <- 1
for (i in sample) {
#for (i in 1:100) {
  if (j %% 100 == 0) print(j)

  train[i,nGramHitRate11 := nGramHitRate(question1,question2,1,1)]
  train[i,nGramHitRate22 := nGramHitRate(question1,question2,2,2)]

  if (length(t1) >0) train[i,lastWordCount1:=word.count[t1[length(t1)]]]
  if (length(t2) >0) train[i,lastWordCount2:=word.count[t2[length(t2)]]]
  
  t1 <- unlist(tokenize_ngrams(train$question1[i],n=2, n_min=2))
  t2 <- unlist(tokenize_ngrams(train$question2[i],n=2, n_min=2))
  sharedPhrases <- intersect(t1,t2)

  bothLengths <- length(t1)+length(t2)
  train[i,nGramHitRate22 := 2*length(sharedPhrases)/bothLengths]
  train[i,nGramHitRate22 := ifelse(is.infinite(nGramHitRate22),0,nGramHitRate22)]

  t1 <- unlist(tokenize_ngrams(train$question1[i],n=3, n_min=3))
  t2 <- unlist(tokenize_ngrams(train$question2[i],n=3, n_min=3))
  sharedPhrases <- intersect(t1,t2)
  
  bothLengths <- length(t1)+length(t2)
  train[i,nGramHitRate33 := 2*length(sharedPhrases)/bothLengths]
  train[i,nGramHitRate33 := ifelse(is.infinite(nGramHitRate33),0,nGramHitRate33)]
  
  t1 <- unlist(tokenize_ngrams(train$question1[i],n=4, n_min=4))
  t2 <- unlist(tokenize_ngrams(train$question2[i],n=4, n_min=4))
  sharedPhrases <- intersect(t1,t2)
  
  bothLengths <- length(t1)+length(t2)
  train[i,nGramHitRate44 := 2*length(sharedPhrases)/bothLengths]
  train[i,nGramHitRate44 := ifelse(is.infinite(nGramHitRate44),0,nGramHitRate44)]
  
  # # # 
  t1 <- unlist(strsplit(tolower(train$question1[i]), " "))
  t2 <- unlist(strsplit(tolower(train$question2[i]), " "))
  # remove stop words
  t1 <- setdiff(t1,stopWords)
  t2 <- setdiff(t2,stopWords)
  sharedWords <- intersect(t1,t2)

  bothLengths <- length(t1)+length(t2)
  train[i,nGramHitRate := 2*length(sharedWords)/bothLengths]
  train[i,nGramHitRate := ifelse(is.infinite(nGramHitRate),0,nGramHitRate)]
  train[i,nCommonWords :=length(sharedWords)]
  train[i,nWordsFirst := length(t1)]
  train[i,nWordsSecond := length(t2)]
  
  # Last word
  if (length(t1)>0 & length(t2)>0) {
    train[i,lastWord := as.integer(t1[length(t1)] == t2[length(t2)])]
  } else 
  {
    train[i,lastWord := 0]
    
  }
  # Last Two Words
  if (length(t1)>1 & length(t2)>1) {
    train[i,lastTwoWords := as.integer(t1[(length(t1)-1)] == t2[(length(t2)-1)] &&
                                         t1[length(t1)] == t2[length(t2)])   ]
  } else 
  {
    train[i,lastWord := 0]
    
  }
  
  #train[i,lastWordCount:=word.count[t1]]
  #word.weight.filter <- word.weight[c(t1,t2)]

  # sharedWeights <- 2*sum(word.weight.filter[sharedWords][!is.na(word.weight.filter[sharedWords])])
  # individualWeights <- sum(word.weight.filter[t1][!is.na(word.weight.filter[t1])])+sum(word.weight.filter[t2][!is.na(word.weight.filter[t2])])
  # train[i,dfIdf:= sharedWeights/individualWeights]
  j = j+1
}

auc(train$is_duplicate[sample],train$nGramHitRate[sample])
auc(train$is_duplicate[sample],train$nGramHitRate22[sample])
auc(train$is_duplicate[sample],train$nGramHitRate33[sample])
auc(train$is_duplicate[sample],train$lastWord[sample])

samplePos <- intersect(which(train$is_duplicate==1),sample)
sampleNeg <- intersect(which(train$is_duplicate==0),sample)
positivePct <- length(samplePos)/length(sample)
negativePct <- (1-positivePct)
positiveTargetPct <- 0.165
negativeTargetPct <- 1- positiveTargetPct

add <- as.integer((negativeTargetPct*sampleSize - length(sampleNeg))/(1-negativeTargetPct))
newSample <- c(samplePos,sample(sampleNeg, length(sampleNeg) + add, replace=TRUE))

train.final <- train[newSample]
s <- sample(1:nrow(train.final),0.8*nrow(train.final))

xgb_params = list(seed = 0,subsample = 0.9,
  eta = 0.1,max_depth =8,num_parallel_tree = 1,min_child_weight = 1,
  objective='binary:logistic',eval_metric = 'logloss')

feature.names <- c("nGramHitRate","nGramHitRate11","nGramHitRate22","nGramHitRate33","nGramHitRate44","lastWord","lastTwoWords",
                   "nCommonWords","nWordsFirst","nWordsSecond")
                   #"lastWordCount1","lastWordCount2")
#feature.names <- c("nGramHitRate")

dtrain = xgb.DMatrix(as.matrix(train.final[s,feature.names,with=FALSE]), label=train.final$is_duplicate[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(train.final[-s,feature.names,with=FALSE]), label=train.final$is_duplicate[-s], missing=NA)
dall = xgb.DMatrix(as.matrix(train.final[,feature.names,with=FALSE]), label=train.final$is_duplicate, missing=NA)

watchlist <- list(train=dtrain,valid=dvalid)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early_stopping_rounds =  100,nrounds=10000,
                          maximize=FALSE,verbose = TRUE,print_every_n=10,watchlist = watchlist)

train.final$pred <- predict(xgboost.fit,dall)
xgb.importance(feature.names,xgboost.fit)

# Mistakes made
train.final[abs(train.final$pred - train.final$is_duplicate) > 0.8]

