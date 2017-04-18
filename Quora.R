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
library(rlist)

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

getNGramMatch <- function(question1,question2,is_duplicate,n=2) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- intersect(t1,t2)
  if (length(sharedPhrases) > 0) {
    ret <- cbind(sharedPhrases,is_duplicate)
  } else {
    ret <- NULL
  }
  ret
}

# Tuple generatio
nTupleDiagnostics <- function(n=2) {
  word.tuples <- mapply(getNGramMatch, train$question1,train$question2,train$is_duplicate,n,USE.NAMES=FALSE, SIMPLIFY=TRUE)
  word.tuples <- list.clean(word.tuples, fun = is.null, recursive = FALSE)
  word.tuples <- list.rbind(word.tuples)
  word.tuples <- as.data.table(word.tuples)
  word.tuples$is_duplicate <- as.numeric(word.tuples$is_duplicate)
  setkeyv(word.tuples,c("sharedPhrases"))
  tuple.mean.duplicate <- word.tuples[,j=list(prob=mean(is_duplicate),count=length(is_duplicate)),by=list(sharedPhrases)]
  tuple.mean.duplicate  
}

nTuple1 <- nTupleDiagnostics(1)
nTuple2 <- nTupleDiagnostics(2)
nTuple3 <- nTupleDiagnostics(3)
nTuple4 <- nTupleDiagnostics(4)

save("nTuple1","nTuple2","nTuple3","nTuple4",file="N-Tuples Quora")
load("N-Tuples Quora")

# t <- t1[t1$count > 100 & t1$prob>0.80]
# sum(t$count)
# term <- 'your new year' # 112
# s <- train[tolower(train$question1) %like% term & tolower(train$question2) %like% term]
# s[1:5]
# st <- test[tolower(test$question1) %like% term & tolower(test$question2) %like% term]
# st[1:5]

# Doodling here
# word.tuples <- c(unlist(sapply(train$question1[1:10],function(x) tokenize_ngrams(x,n=2,n_min=2,simplify = FALSE)),use.names = FALSE),
#                        unlist(sapply(train$question2[1:10],function(x) tokenize_ngrams(x,n=2,n_min=2)),use.names = FALSE))
# 
# word.tuples <- table(c(unlist(sapply(train$question1,function(x) tokenize_ngrams(x,n=2,n_min=2)),use.names = FALSE),
#                        unlist(sapply(train$question2,function(x) tokenize_ngrams(x,n=2,n_min=2)),use.names = FALSE)))
# word.tuples.test <- table(c(unlist(sapply(test$question1[1:400000],function(x) tokenize_ngrams(x,n=2,n_min=2)),use.names = FALSE),
#                        unlist(sapply(test$question2[1:400000],function(x) tokenize_ngrams(x,n=2,n_min=2)),use.names = FALSE)))
# print("intersect")
# length(intersect(names(word.tuples.test[word.tuples.test>5000]),names(word.tuples[word.tuples>5000])))
# print("untion")
# length(union(names(word.tuples.test[word.tuples.test>5000]),names(word.tuples[word.tuples>5000])))
# 
# word.triples <- table(c(unlist(sapply(train$question1,function(x) tokenize_ngrams(x,n=3,n_min=3)),use.names = FALSE),
#                        unlist(sapply(train$question2,function(x) tokenize_ngrams(x,n=3,n_min=3)),use.names = FALSE)))

# save("word.tuples","word.triples",file="Word Combinations Quora")
# load("Word Combinations Quora")

# word.tuples.most.common <- word.tuples[word.tuples>500]
# word.comb.features <- as.vector(NULL)
# for (tuple in 1:nrow(word.tuples.most.common)) {
#     term <- names(word.tuples.most.common[tuple])
#     print(paste0("Tuple : ",tuple, " of ",nrow(word.tuples.most.common)))
#     print(paste0("The term is : ",term))  
#     tuple.mean <- mean(train$is_duplicate[grepl(term,tolower(train$question2)) & grepl(term,tolower(train$question1))])
#     tuple.sum <- sum(train$is_duplicate[grepl(term,tolower(train$question2)) & grepl(term,tolower(train$question1))])
#     # add the feature
#     if (tuple.mean > 0.7 & !is.nan(tuple.mean)) {
#       feature.name <- paste0("feature_",paste0(unlist(strsplit(term," ")),collapse="_"))
#       word.comb.features <- c(word.comb.features,feature.name)
#       print(paste0("The mean is: ",tuple.mean," and the sum duplicates are ",tuple.sum))
#       train[,eval(feature.name) := as.numeric(grepl(term,tolower(train$question2)) & grepl(term,tolower(train$question1)))]
#     }
# }

#train[grepl(term,tolower(train$question2)) & grepl(term,tolower(train$question1))]

#dfCorpus = Corpus(VectorSource(c(train$question1[-sample],train$question2[-sample])))
#inspect(dfCorpus
#myTdm <- TermDocumentMatrix(dfCorpus, control=list(tolower=TRUE))
#word.count <- slam::row_sums(myTdm)
#word.weight <- sapply(word.count, function(x) getWordWeights(x,5000,2))
stopWords <- tm::stopwords(kind="en")

wordVecSpace <- read.binary.vectors("GoogleNews-vectors-negative300.bin",nrows=100000)
words.google <-  attr(wordVecSpace, which="dimnames")[[1]]

TupleProb <- function(question1,question2,n=2,nTupleTable=NULL) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- intersect(t1,t2)
  idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases)
  ret.prob <-sum(nTupleTable$prob[idx] * nTupleTable$count[idx])/sum(nTupleTable$count[idx])
  ret.count <-sum(nTupleTable$count[idx])
  list(prob=ret.prob,count=ret.count)
}

nGramHitRate <- function(question1,question2,n=1,n_min=1,googleDictMin=500,stemming=TRUE) {
  if (stemming==TRUE) {
    question1 <- paste0(tokenize_word_stems(question1,language="en", simplify=TRUE),collapse=" ")
    question2 <- paste0(tokenize_word_stems(question2,language="en", simplify=TRUE),collapse=" ")
  }
  t1 <- unlist(tokenize_ngrams(question1,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  t2 <- unlist(tokenize_ngrams(question2,n=n, n_min=n_min,stopwords=words.google[1:googleDictMin]))
  sharedPhrases <- intersect(t1,t2)
  bothLengths <- length(t1)+length(t2)
  hitRate <- 2*length(sharedPhrases)/bothLengths
  hitRate <- ifelse(is.infinite(hitRate) | is.nan(hitRate),0,hitRate)
  hitRate
}

sentenceOverlap <- function(question1,question2,n=20,stopWords,stemming=TRUE) {
  # Do stemming first
  question1 <- paste0(tokenize_word_stems(question1,language="en", simplify=TRUE),collapse=" ")
  question2 <- paste0(tokenize_word_stems(question2,language="en", simplify=TRUE),collapse=" ")
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
  #ret <- overlapSum
  ret <- overlapSum/(q1Length+q2Length)
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
sampleSize <- 10000
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
  
  train[i,sentenceOverlap:= sentenceOverlap(question1,question2,20,"")]
  
  ret <- TupleProb(train$question1[i],train$question2[i],n=3,nTupleTable=nTuple3)
  train[i,Tuple3Prob:= ret$prob]
  train[i,Tuple3Count:= ret$count]
  
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
  ret <- TupleProb(train$question1[i],train$question2[i],n=3,nTupleTable=nTuple3)
  train[i,Tuple3Prob:= ret$prob]
  train[i,Tuple3Count:= ret$count]
  
  ret <- TupleProb(train$question1[i],train$question2[i],n=2,nTupleTable=nTuple2)
  train[i,Tuple2Prob:= ret$prob]
  train[i,Tuple2Count:= ret$count]
  
  ret <- TupleProb(train$question1[i],train$question2[i],n=1,nTupleTable=nTuple1)
  train[i,Tuple1Prob:= ret$prob]
  train[i,Tuple1Count:= ret$count]
  
  ret <- TupleProb(train$question1[i],train$question2[i],n=4,nTupleTable=nTuple4)
  train[i,Tuple4Prob:= ret$prob]
  train[i,Tuple4Count:= ret$count]
  
  j = j+1
}

xgb_params = list(seed = 0,subsample = 1,
  eta = 0.1,max_depth =4,num_parallel_tree = 1,min_child_weight = 2,
  objective='binary:logistic',eval_metric = 'logloss')

feature.names <- c("nGramHitRate",
                   "nGramHitRate11","nGramHitRate22","nGramHitRate33",
                   "nGramHitRate11_0","nGramHitRate22_0","nGramHitRate33_0","nGramHitRate33_0",
                   "sentenceOverlap",
                   "q1Minusq2","q2Minusq1",
                   "nGramSkipHitRate21","nGramSkipHitRate31","nGramSkipHitRate41","nGramSkipHitRate42",
                   "nCommonWords","nWordsFirst","nWordsSecond","firstWordEqual",
                   "Tuple3Prob","Tuple3Count","Tuple4Prob","Tuple4Count",
                   "Tuple2Prob","Tuple2Count","Tuple1Prob","Tuple1Count"
                   )
#feature.names <- union(feature.names,word.comb.features)

dtrain = xgb.DMatrix(as.matrix(train[s,feature.names,with=FALSE]), label=train$is_duplicate[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(train[v,feature.names,with=FALSE]), label=train$is_duplicate[v], missing=NA)
dall = xgb.DMatrix(as.matrix(train[sample,feature.names,with=FALSE]), label=train$is_duplicate[sample], missing=NA)

watchlist <- list(train=dtrain,valid=dvalid)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early_stopping_rounds =  50,nrounds=10000,
                          maximize=FALSE,verbose = TRUE,print_every_n=10,watchlist = watchlist)

train$pred[sample] <- predict(xgboost.fit,dall)
xgb.importance(feature.names,xgboost.fit)

# check 23015, how to improve
train[23015]

# save model and data
save(list=c("xgboost.fit","feature.names","train","s","v","sample"), file="Quora Save Model")
load("Quora Save Model")

# Mistakes made
train.final <- train[sample]
train.final[abs(train.final$pred - train.final$is_duplicate) > 0.6][1:10]
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
