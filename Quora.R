library(data.table)
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

# Create a corpus without the data
sampleSize <- 10000
sample <- sample(1:nrow(train),sampleSize)
print(length(sample))

dfCorpus = Corpus(VectorSource(c(train$question1[-sample],train$question2[-sample])))
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

lastWordCount <- function(question) {
  t <- unlist(tokenize_ngrams(question,n=1, n_min=1))
  if (length(t)==0) 0 else {
    #word.count[t[length(t)]]  %/% 100
    word.count[t[length(t)]]/sum(word.count)*100
  }
}

lastTwoWordCount <- function(question) {
  t <- unlist(tokenize_ngrams(question,n=1, n_min=1))
  if (length(t)<2) 0 else {
    #word.count[t[length(t)]]  %/% 100
    word1.count <- word.count[t[length(t)]]
    word2.count <- word.count[t[length(t)-1]]
    mean(c(word1.count,word2.count),na.rm=TRUE)/sum(word.count)*100/2
  }
}

sharedWordsLastN <- function(question1,question2,n=1,stopWords) {
  t1 <- unlist(strsplit(tolower(question1), " "))
  t2 <- unlist(strsplit(tolower(question2), " "))
  # remove stop words
  t1 <- setdiff(t1,stopWords)
  t2 <- setdiff(t2,stopWords)
  if (length(t1)>0 & length(t2)>0) {
    t1.idx <- max(length(t1)-n+1,1):length(t1)
    t2.idx <- max(length(t2)-n+1,1):length(t2)
    length(intersect(t1[t1.idx],t2[t2.idx]))
  } else 0
}

j <- 1
for (i in sample) {
#for (i in 1:100) {
  if (j %% 100 == 0) print(j)

  train[i,nGramHitRate11 := nGramHitRate(question1,question2,1,1)]
  train[i,nGramHitRate22 := nGramHitRate(question1,question2,2,2)]
  train[i,nGramHitRate33 := nGramHitRate(question1,question2,3,3)]
  train[i,nGramHitRate44 := nGramHitRate(question1,question2,4,4)]
  train[i,nGramHitRate55 := nGramHitRate(question1,question2,5,5)]
  train[i,nGramHitRate53 := nGramHitRate(question1,question2,5,3)]
  
  train[i,lastWordCount1:=lastWordCount(question1)]
  train[i,lastWordCount2:=lastWordCount(question2)]
  train[i,lastTwoWordCount1:=lastTwoWordCount(question1)]
  train[i,lastTwoWordCount2:=lastTwoWordCount(question2)]
  
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
  train[i,nCommonWordsLastThree := sharedWordsLastN(question1,question2,3,stopWords)]
  
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

# auc(train$is_duplicate[sample],train$nGramHitRate[sample])
# auc(train$is_duplicate[sample],train$nGramHitRate22[sample])
# auc(train$is_duplicate[sample],train$nGramHitRate33[sample])
# auc(train$is_duplicate[sample],train$lastWord[sample])

samplePos <- intersect(which(train$is_duplicate==1),sample)
sampleNeg <- intersect(which(train$is_duplicate==0),sample)
positivePct <- length(samplePos)/length(sample)
negativePct <- (1-positivePct)
positiveTargetPct <- 0.165
#positiveTargetPct <- 0.36
negativeTargetPct <- 1- positiveTargetPct

# Oversample the negatives, but keep track of them 
add <- as.integer((negativeTargetPct*sampleSize - length(sampleNeg))/(1-negativeTargetPct))
sampleNeg <- sample(sampleNeg, length(sampleNeg), replace=TRUE)
sampleNegAdd <- sample(sampleNeg, add, replace=TRUE)

#train.final <- train[newSample]
#train.final <- train[sample]
s <- sample(1:nrow(train.final),0.8*nrow(train.final))

xgb_params = list(seed = 0,subsample = 0.8,
  eta = 0.1,max_depth =4,num_parallel_tree = 1,min_child_weight = 2,
  objective='binary:logistic',eval_metric = 'logloss')

feature.names <- c("nGramHitRate","nGramHitRate11","nGramHitRate22","nGramHitRate33",
                   "nGramHitRate44","nGramHitRate55",#"nGramHitRate53",
                   "lastWord","lastTwoWords","nCommonWordsLastThree",
                   "nCommonWords","nWordsFirst","nWordsSecond",
                   "lastWordCount1","lastWordCount2","lastTwoWordCount1","lastTwoWordCount2"
                   )
#feature.names <- c("nGramHitRate")

dtrain = xgb.DMatrix(as.matrix(train.final[s,feature.names,with=FALSE]), label=train.final$is_duplicate[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(train.final[-s,feature.names,with=FALSE]), label=train.final$is_duplicate[-s], missing=NA)
dall = xgb.DMatrix(as.matrix(train.final[,feature.names,with=FALSE]), label=train.final$is_duplicate, missing=NA)

watchlist <- list(train=dtrain,valid=dvalid)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early_stopping_rounds =  10,nrounds=10000,
                          maximize=FALSE,verbose = TRUE,print_every_n=10,watchlist = watchlist)

train.final$pred <- predict(xgboost.fit,dall)
xgb.importance(feature.names,xgboost.fit)

# Mistakes made
train.final[abs(train.final$pred - train.final$is_duplicate) > 0.8]

