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

setwd <- "C:/R-Studio/Quora"
source('Quora_Functions.R')
train  <- fread( "./train.csv")
test  <- fread( "./test.csv")
train$is_duplicate <- as.numeric(train$is_duplicate)

stopWords <- tm::stopwords(kind="en")
# wordVecSpace <- read.binary.vectors("GoogleNews-vectors-negative300.bin",nrows=500000)
# words.google <-  attr(wordVecSpace, which="dimnames")[[1]]
# save("words.google",file="Quora Google Dictionary")
load("Quora Google Dictionary")
load("Quora Save Model")

trainNGramFileName <- "Quora NGrams Train"
sampleSize <- 50000
sample <- sample(1:nrow(train),sampleSize)
tupleSample <- setdiff(1:nrow(train),sample)
#tupleSample <- 1:1000
calcAndSaveNGramTables(tupleSample, fileName = trainNGramFileName)

load(trainNGramFileName)
#sample <- sample(sample,50000)

train <- traintestAddColumn(sample,train)
  
samplePos <- intersect(which(train$is_duplicate==1),sample)
sampleNeg <- intersect(which(train$is_duplicate==0),sample)
positivePct <- length(samplePos)/length(sample)
negativePct <- (1-positivePct)
positiveTargetPct <- 0.165
#positiveTargetPct <- 0.36
negativeTargetPct <- 1- positiveTargetPct

# Oversample the negatives, but keep track of them 
add <- as.integer((negativeTargetPct*length(sample) - length(sampleNeg))/(1-negativeTargetPct))
sampleNeg <- sample(sampleNeg, length(sampleNeg)+add, replace=TRUE)
sampleNeg <- sampleNeg[order(sampleNeg)]

s <- c( samplePos[1:(0.8*length(samplePos))], sampleNeg[1:(length(sampleNeg)*0.8)])
v <- c( samplePos[(as.integer(0.8*length(samplePos)+1)):length(samplePos)],
        sampleNeg[(as.integer(0.8*length(sampleNeg)+1)):length(sampleNeg)])

#Additional columns
j <- 1
for (i in sample) {
  print(j)
  
  ret <- WordMissProbTopN(train$question1[i],train$question2[i],n=2,nTupleTable=nWordMiss2,countMin = 3, TopN = 1, decreasing=FALSE )
  train[i,WordMiss2ProbTop1Min3:= ret$prob]
  train[i,WordMiss2CountTop1Min3:= ret$count]
  
  j = j+1
}
idx <- which(!is.na(train$WordMiss2ProbTop1[sample]))
cor(train$WordMiss2ProbTop1[sample[idx]],train$is_duplicate[sample[idx]])

feature.names <- c("nGramHitRate",
                   "nGramHitRate11_0","nGramHitRate22_0",
                   "Tuple1ProbTop1Min","Tuple1CountTop1Min",
                   "Tuple2ProbTop1Min","Tuple2CountTop1Min",
                   "WordMatch2ProbTop1","WordMatch2CountTop1",
                   "WordMatch3ProbTop1","WordMatch3CountTop1",
                   "WordMatch2ProbTop1Stem","WordMatch2CountTop1Stem",
                   "WordMiss2ProbTop1Min3","WordMiss2CountTop1Min3",
                   "WordMiss2ProbTop1","WordMiss2CountTop1",
                   "WordMiss2ProbTop2","WordMiss2CountTop2",
                   "WordMiss2ProbTop1Max","WordMiss2CountTop1Max",
                   #"WordMiss2ProbTop1Stem","WordMiss2CountTop1Stem",
                   "Tuple1ProbMax","Tuple1CountMax")

col.names <- c("id","qid1","qid2","question1","question2","is_duplicate")
col.names <- c(col.names,feature.names)
#train <- train[,col.names,with=FALSE]

xgb_params = list(seed = 0,subsample = 0.8,eta = 0.1,max_depth =3,num_parallel_tree = 1,min_child_weight = 2,
                  objective='binary:logistic',eval_metric = 'logloss')

dtrain = xgb.DMatrix(as.matrix(train[s,feature.names,with=FALSE]), label=train$is_duplicate[s], missing=NA)
dvalid = xgb.DMatrix(as.matrix(train[v,feature.names,with=FALSE]), label=train$is_duplicate[v], missing=NA)
dall = xgb.DMatrix(as.matrix(train[sample,feature.names,with=FALSE]), label=train$is_duplicate[sample], missing=NA)

watchlist <- list(train=dtrain,valid=dvalid)
xgboost.fit <- xgb.train (data=dtrain,xgb_params,missing=NA,early_stopping_rounds =  50,nrounds=10000,
                          maximize=FALSE,verbose = TRUE,print_every_n=10,watchlist = watchlist)

train$pred[sample] <- predict(xgboost.fit,dall)
xgb.importance(feature.names,xgboost.fit)

# save model and data
save(list=c("xgboost.fit","feature.names","train","s","v","sample"), file="Quora Save Model")

load("Quora Save Model")

# Mistakes made
train.final <- train[sample]
train.final[abs(train.final$pred - train.final$is_duplicate) > 0.70][20:30]
t <- train.final[abs(train.final$pred - train.final$is_duplicate) < 0.1]
t[t$is_duplicate==1]


idx <- which(train$WordMiss2CountTop1>4)
cor(train$WordMiss2ProbTop1[idx], train$is_duplicate[idx])

# Read wrod2vec files
# library(wordVectors)
# wordVecSpace <- read.binary.vectors("GoogleNews-vectors-negative300.bin",nrows=200000)
# words <-  attr(wordVecSpace, which="dimnames")[[1]]
# word.intersect <- intersect(names(word.count),tolower(words)) 
# length(word.intersect)
# print(paste0("coverage is ",sum(word.count[word.intersect])/sum(word.count)*100))
# 
# length(setdiff(names(word.count),tolower(words)))
# 
# setdiff(names(word.count),tolower(words))[9000:10000]
# words[which(stringdist("delivary",words)==1)[1]]
# train[which(train$question1 %like% 'delivary')]

# Train an own model
# write.table(c(train$question1[1:10000],train$question2[1:10000]),"Question Text File.txt",quote=FALSE,col.names=FALSE,row.names=FALSE)
# vs <- train_word2vec("Question Text File.txt",output_file="vectors.bin",vectors=50,force=TRUE)
# closest_to(vs,"good",n=5)

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
