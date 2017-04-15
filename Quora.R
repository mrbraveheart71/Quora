library(data.table)
library(xgboost)
library(stringi)
library(tm)
library(text2vec)
library(wordnet)
library(tokenizers)
library(xgboost)
library(pROC)
library(stringdist)
library(wordVectors)
library(koRpus)
library(SnowballC)
library(coreNLP)

#initCoreNLP(libLoc = "C:/R-Studio/Quora/stanford-corenlp-full-2016-10-31",type="english")

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

#dfCorpus = Corpus(VectorSource(c(train$question1[-sample],train$question2[-sample])))
#inspect(dfCorpus
#myTdm <- TermDocumentMatrix(dfCorpus, control=list(tolower=TRUE))

#word.count <- slam::row_sums(myTdm)
#word.weight <- sapply(word.count, function(x) getWordWeights(x,5000,2))
  
# seldomWords <- names(word.count[word.count < 15000])
# frequentWords <- names(word.count[word.count >= 15000])
stopWords <- tm::stopwords(kind="en")

wordVecSpace <- read.binary.vectors("GoogleNews-vectors-negative300.bin",nrows=100000)
words.google <-  attr(wordVecSpace, which="dimnames")[[1]]

sentenceOverlap <- function(question1,question2,n=10,stopWords) {
  t1 <- tokenize_ngrams(question1,n=n, n_min=1,stopwords=stopWords,simplify=TRUE)
  q1Length <- length(tokenize_words(question1,stopwords = stopWords,simplify=TRUE))
  t2 <- tokenize_ngrams(question2,n=n, n_min=1,stopwords=stopWords,simplify =TRUE)
  q2Length <- length(tokenize_words(question2,stopwords = stopWords,simplify=TRUE))
  overlap <- intersect(t1,t2)
  if (length(overlap>0)) { 
    overlapSum <- sum(sapply(overlap,function(x) length(tstrsplit(x," "))^2))
  } else {
    overlapSum <- 0
  }
  #ret <- tanh(overlapSum/(q1Length+q2Length))
  ret <- overlapSum
}

nGramSkipHitRate <- function(question1,question2,n=2,k=1) {
  t1 <- tokenize_skip_ngrams(question1,n=n, k=k, simplify = TRUE)
  t2 <- tokenize_skip_ngrams(question2,n=n, k=k, simplify = TRUE)
  sharedPhrases <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  hitRate <- 2*length(sharedPhrases)/bothLengths
  hitRate <- ifelse(is.infinite(hitRate),0,hitRate)
  hitRate
}

nGramHitRate <- function(question1,question2,n=1,n_min=1,googleDictMin=500) {
  t1 <- unlist(tokenize_ngrams(question1,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  t2 <- unlist(tokenize_ngrams(question2,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  sharedPhrases <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  hitRate <- 2*length(sharedPhrases)/bothLengths
  hitRate <- ifelse(is.infinite(hitRate) | is.nan(hitRate),0,hitRate)
  hitRate
}


nGramHitRateStem <- function(question1,question2,n=1,n_min=1,googleDictMin=500) {
  t1 <- tokenize_ngrams(question1,n=1, n_min=1,stopwords=words.google[1:googleDictMin], simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=1, n_min=1,stopwords=words.google[1:googleDictMin], simplify = TRUE)
  t1 <- as.vector(sapply(t1,function(x) wordStem(x,language = "english")))
  t2 <- as.vector(sapply(t2,function(x) wordStem(x,language = "english")))
  sharedPhrases <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  hitRate <- 2*length(sharedPhrases)/bothLengths
  hitRate <- ifelse(is.infinite(hitRate) | is.nan(hitRate),0,hitRate)
  hitRate
}

nGramHitCount <- function(question1,question2,n=1,n_min=1,googleDictMin=500) {
  t1 <- unlist(tokenize_ngrams(question1,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  t2 <- unlist(tokenize_ngrams(question2,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  sharedPhrases <- intersect(t1,t2)
  hitRate <- length(sharedPhrases)
}


lastNWordsPunct <- function(question1,question2,n=2, googleDictMin=500) {
  t1 <- unlist(strsplit(tolower(question1), " "))
  t2 <- unlist(strsplit(tolower(question2), " "))
  # remove stop words
  t1 <- setdiff(t1,words.google[1:googleDictMin])
  t2 <- setdiff(t2,words.google[1:googleDictMin])

  t1.start <- max(1,length(t1)-n+1)
  t2.start <- max(1,length(t2)-n+1)
  hitRate <- length(intersect(t1[t1.start:length(t1)],t2[t2.start:length(t2)]))
  hitRate
}


sharedWords <- function(question1,question2,googleDictMin=500) {
  t1 <- unlist(tokenize_ngrams(question1,n=1, n_min=1,stopwords=words.google[1:googleDictMin]))
  t2 <- unlist(tokenize_ngrams(question2,n=1, n_min=1,stopwords=words.google[1:googleDictMin]))
  sharedPhrases <- intersect(t1,t2)
  length(sharedPhrases)
}

sharedApproximateWords <- function(question1,question2,googleDictMin=500, approx=3) {
  t1 <- unlist(tokenize_ngrams(question1,n=1, n_min=1,stopwords=words.google[1:googleDictMin]))
  t1 <- unique(as.vector(sapply(t1, function(x) closest_to(wordVecSpace,x,approx)$word)))
  t2 <- unlist(tokenize_ngrams(question2,n=1, n_min=1,stopwords=words.google[1:googleDictMin]))
  t2 <- unique(as.vector(sapply(t2, function(x) closest_to(wordVecSpace,x,approx)$word)))
  sharedPhrases <- intersect(t1,t2)
  length(sharedPhrases)/(0.5*length(t1)+0.5*length(t2))
}

lastWordCount <- function(question,stopwords=NULL) {
  t <- unlist(tokenize_ngrams(question,n=1, n_min=1,stopwords=stopWords))
  if (length(t)==0) 0 else {
    #word.count[t[length(t)]]  %/% 100
    word.count[t[length(t)]]/sum(word.count)*100
  }
}

lastTwoWordCount <- function(question, stopwords=NULL) {
  t <- unlist(tokenize_ngrams(question,n=1, n_min=1,stopwords=stopWords))
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


# Create a corpus without the data
sampleSize <- 30000
sample <- sample(1:nrow(train),sampleSize)
print(length(sample))

j <- 1
for (i in sample) {
#for (i in 1:100) {
  if (j %% 10 == 0) print(j)

  train[i,nGramHitRate11 := nGramHitRate(question1,question2,1,1,500)]
  train[i,nGramHitRate22 := nGramHitRate(question1,question2,2,2,500)]
  train[i,nGramHitRate33 := nGramHitRate(question1,question2,3,3,500)]

  train[i,nGramHitRate11_0 := nGramHitRate(question1,question2,1,1,0)]
  train[i,nGramHitRate22_0 := nGramHitRate(question1,question2,2,2,0)]
  train[i,nGramHitRate33_0 := nGramHitRate(question1,question2,3,3,0)]
  #train[i,nGramHitRate44_0 := nGramHitRate(question1,question2,4,4,0)]
  #train[i,nGramHitRate55_0 := nGramHitRate(question1,question2,5,5,0)]
  
  # train[i,nGramHitRate44 := nGramHitRate(question1,question2,4,4,500)]
  # train[i,nGramHitRate55 := nGramHitRate(question1,question2,5,5,500)]
  # 
  # train[i,nGramHitRate11_5000 := nGramHitRate(question1,question2,1,1,5000)]
  # train[i,nGramHitRate22_5000 := nGramHitRate(question1,question2,2,2,5000)]
  # train[i,nGramHitRate33_5000 := nGramHitRate(question1,question2,3,3,5000)]

  train[i,nGramSkipHitRate21 := nGramSkipHitRate(question1,question2,2,1)]
  train[i,nGramSkipHitRate31 := nGramSkipHitRate(question1,question2,3,1)]
  train[i,nGramSkipHitRate41 := nGramSkipHitRate(question1,question2,4,1)]
  train[i,nGramSkipHitRate42 := nGramSkipHitRate(question1,question2,4,2)]
  
  # train[i,lastWordCount1:=lastWordCount(question1)]
  # train[i,lastWordCount2:=lastWordCount(question2)]

  # # # 
  t1 <- unlist(strsplit(tolower(train$question1[i]), " "))
  t2 <- unlist(strsplit(tolower(train$question2[i]), " "))
  train[i,firstWordEqual := ifelse(t1[1]==t2[1],1,0)]
  # remove stop words
  t1 <- setdiff(t1,stopWords)
  t2 <- setdiff(t2,stopWords)
  sharedWords <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  train[i,nGramHitRate := 2*length(sharedWords)/bothLengths]
  train[i,nGramHitRate := ifelse(is.infinite(nGramHitRate),0,nGramHitRate)]
  
  train[i,nCommonWords :=length(unique(sharedWords))]
  train[i,nWordsFirst := length(unique(t1))]
  train[i,nWordsSecond := length(unique(t2))]
  
  t1t <- unlist(tokenize_ngrams(train$question1[i],n=1, n_min=1))
  t2t <- unlist(tokenize_ngrams(train$question2[i],n=1, n_min=1))
  train[i, q1Minusq2 := length(setdiff(t1t,t2t))]
  train[i, q2Minusq1 := length(setdiff(t2t,t1t))]

  j = j+1
}


samplePos <- intersect(which(train$is_duplicate==1),sample)
sampleNeg <- intersect(which(train$is_duplicate==0),sample)
positivePct <- length(samplePos)/length(sample)
negativePct <- (1-positivePct)
positiveTargetPct <- 0.165
#positiveTargetPct <- 0.36
negativeTargetPct <- 1- positiveTargetPct

# Oversample the negatives, but keep track of them 
add <- as.integer((negativeTargetPct*sampleSize - length(sampleNeg))/(1-negativeTargetPct))
sampleNeg <- sample(sampleNeg, length(sampleNeg)+add, replace=TRUE)
sampleNeg <- sampleNeg[order(sampleNeg)]

s <- c( samplePos[1:(0.8*length(samplePos))], sampleNeg[1:(length(sampleNeg)*0.8)])
v <- c( samplePos[(as.integer(0.8*length(samplePos)+1)):length(samplePos)],
        sampleNeg[(as.integer(0.8*length(sampleNeg)+1)):length(sampleNeg)])

#Additional columns
j <- 1
for (i in sample) {
  print(j)
  #train[i,sentenceOverlap100:= sentenceOverlap(question1,question2,20,100)]
  train[i,sentenceOverlap:= sentenceOverlap(question1,question2,20,"")]
  
  #   train[i,lastTwoWordCount1:=lastTwoWordCount(question1)]
  #   train[i,lastTwoWordCount2:=lastTwoWordCount(question2)]

  j = j+1
}

xgb_params = list(seed = 0,subsample = 1,
  eta = 0.1,max_depth =6,num_parallel_tree = 1,min_child_weight = 2,
  objective='binary:logistic',eval_metric = 'logloss')

feature.names <- c("nGramHitRate",
                   #"nGramHitRate11","nGramHitRate22","nGramHitRate33",#"nGramHitRate44","nGramHitRate55",
                   #"nGramHitRate11_0","nGramHitRate22_0","nGramHitRate33_0","nGramHitRate33_0",#"nGramHitRate55_0",
                   #"q1Minusq2",
                   "sentenceOverlap"
                   #"q1Minusq2","q2Minusq1",
                   #"nGramSkipHitRate21","nGramSkipHitRate31","nGramSkipHitRate41","nGramSkipHitRate42",
                   #"nCommonWords","nWordsFirst","nWordsSecond","firstWordEqual"
                   )

dtrain = xgb.DMatrix(as.matrix(train[s,feature.names,with=FALSE]), label=train$is_duplicate[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(train[v,feature.names,with=FALSE]), label=train$is_duplicate[v], missing=NA)
dall = xgb.DMatrix(as.matrix(train[sample,feature.names,with=FALSE]), label=train$is_duplicate[sample], missing=NA)

watchlist <- list(train=dtrain,valid=dvalid)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early_stopping_rounds =  50,nrounds=10000,
                          maximize=FALSE,verbose = TRUE,print_every_n=10,watchlist = watchlist)

train$pred[sample] <- predict(xgboost.fit,dall)
xgb.importance(feature.names,xgboost.fit)

# check 97186, something wrong

# save model and data
save(list=c("xgboost.fit","feature.names","train","s","v","sample","word.count","words"), file="Quora Save Model")
load("Quora Save Model")

# Mistakes made
train.final <- train[sample]
train.final[abs(train.final$pred - train.final$is_duplicate) > 0.9]
t <- train.final[abs(train.final$pred - train.final$is_duplicate) < 0.1]
t[t$is_duplicate==1]


# Read wrod2vec files
library(wordVectors)
wordVecSpace <- read.binary.vectors("GoogleNews-vectors-negative300.bin",nrows=200000)
words <-  attr(wordVecSpace, which="dimnames")[[1]]
word.intersect <- intersect(names(word.count),tolower(words)) 
length(word.intersect)
print(paste0("coverage is ",sum(word.count[word.intersect])/sum(word.count)*100))

length(setdiff(names(word.count),tolower(words)))

setdiff(names(word.count),tolower(words))[9000:10000]
words[which(stringdist("delivary",words)==1)[1]]
train[which(train$question1 %like% 'delivary')]

# Train an own model
write.table(c(train$question1[1:10000],train$question2[1:10000]),"Question Text File.txt",quote=FALSE,col.names=FALSE,row.names=FALSE)
vs <- train_word2vec("Question Text File.txt",output_file="vectors.bin",vectors=50,force=TRUE)
closest_to(vs,"good",n=5)
