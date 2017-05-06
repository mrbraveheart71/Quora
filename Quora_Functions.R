

MultiLogLoss <- function(act, pred){
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

getWordWeights <- function(count,eps=5000,min_count=2) {
  if (count < min_count) {0} else
  {1 / (count + eps)}
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

generate_ngrams_batch <- function(documents_list, ngram_min, ngram_max, stopwords = character(), ngram_delim = " ") {
  .Call('tokenizers_generate_ngrams_batch', PACKAGE = 'tokenizers', documents_list, ngram_min, ngram_max, stopwords, ngram_delim)
}

getNGramMatchPunct <- function(question1,question2,is_duplicate,n=2) {
  t1 <- strsplit(tolower(question1), " ")
  t2 <- strsplit(tolower(question2), " ")
  t1 <- unlist(generate_ngrams_batch(t1,ngram_min =n,ngram_max=n))
  t2 <- unlist(generate_ngrams_batch(t2,ngram_min =n,ngram_max=n))
  sharedPhrases <- intersect(t1,t2)
  if (length(sharedPhrases) > 0) {
    ret <- cbind(sharedPhrases,is_duplicate)
  } else {
    ret <- NULL
  }
  ret
}

getNGramMiss <- function(question1,question2,is_duplicate,n=2) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- union(setdiff(t1,t2),setdiff(t2,t1))
  if (length(sharedPhrases) > 0) {
    ret <- cbind(sharedPhrases,is_duplicate)
  } else {
    ret <- NULL
  }
  ret
}

getNWordMatch <- function(question1,question2,is_duplicate,n=2, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
    t2 <- wordStem(t2)
  }
  sharedWords <- intersect(t1,t2)
  if (length(sharedWords) > (n-1)) {
    sharedPhrases <- t(combn(sharedWords,n))
    sharedPhrases <- t(apply(sharedPhrases,1,function(x) x[order(x)]))
    sharedPhrases <- apply(sharedPhrases,1,function(x) paste0(x,collapse=" "))
    ret <- cbind(sharedPhrases,is_duplicate)
    colnames(ret)[1] <- "sharedPhrases"
  } else {
    ret <- NULL
  }
  ret
}

getNWordMiss <- function(question1,question2,is_duplicate,n=2, stemming=FALSE,stopwords=FALSE) {
  sharedPhrases1 <- NULL
  sharedPhrases2 <- NULL
  st <- ""
  if (stopwords==TRUE) st <- stopWords
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE, stopwords = st)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
  }
  if (length(t1) > (n-1) ) {
    sharedPhrases1 <- t(combn(t1,n))
    sharedPhrases1 <- t(apply(sharedPhrases1,1,function(x) x[order(x)]))
    sharedPhrases1 <- apply(sharedPhrases1,1,function(x) paste0(x,collapse=" "))
  }
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE,stopwords = st)
  if (stemming==TRUE) {
    t2 <- wordStem(t2)
  }
  if (length(t2) >  (n-1)) {
    sharedPhrases2 <- t(combn(t2,n))
    sharedPhrases2 <- t(apply(sharedPhrases2,1,function(x) x[order(x)]))
    sharedPhrases2 <- apply(sharedPhrases2,1,function(x) paste0(x,collapse=" "))
  }
  if (length(sharedPhrases1>0) & length(sharedPhrases2>0)) {
    sharedPhrases <- union(setdiff(sharedPhrases1,sharedPhrases2),setdiff(sharedPhrases2,sharedPhrases1))
    if (length(sharedPhrases)>0) {
    ret <- cbind(sharedPhrases,is_duplicate)
    colnames(ret)[1] <- "sharedPhrases"
    } else {
      ret <- NULL
    }
  } else {
    ret <- NULL
  }
  ret
}

WordCrossProbTopN <- function(question1,question2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=TRUE, stopwords=stopWords) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE, stopwords = stopwords)
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE, stopwords = stopwords)
  sharedWords <- expand.grid(t1,t2)
  
  if (length(sharedWords) > 0) {
    sharedWords <- t(apply(sharedWords,1,function(x) x[order(x)]))
    sharedWords <- apply(sharedWords,1,function(x) paste0(x,collapse=" "))
    matches <- nTupleTable[sharedWords]
    matches <- matches[!is.na(matches$count)]
    matches <- matches[order(prob,decreasing=decreasing)]
    #idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases & nTupleTable$count > countMin)
    ret.prob <-matches[matches$count > countMin]$prob[1:TopN]
    ret.count <-matches[matches$count > countMin]$count[1:TopN]
  } else {
    ret.prob <- NA 
    ret.count <- NA 
  }
  list(prob=ret.prob,count=ret.count)
}

getCrossMatch <- function(question1,question2,is_duplicate,stopwords) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE, stopwords = stopwords)
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE, stopwords = stopwords)
  sharedWords <- expand.grid(t1,t2)
  if (length(sharedWords) > 0) {
    sharedWords <- t(apply(sharedWords,1,function(x) x[order(x)]))
    sharedWords <- apply(sharedWords,1,function(x) paste0(x,collapse=" "))
    ret <- cbind(sharedWords,is_duplicate)
    colnames(ret)[1] <- "sharedPhrases"
  } else {
    ret <- NULL
  }
  ret
}

nCrossQuestionDiagnostics <- function(train, stopwords=TRUE) {
  word.tuples <- mapply(getCrossMatch, train$question1,train$question2,train$is_duplicate,stopwords,USE.NAMES=FALSE, SIMPLIFY=TRUE)
  word.tuples <- list.clean(word.tuples, fun = is.null, recursive = FALSE)
  word.tuples <- list.rbind(word.tuples)
  word.tuples <- as.data.table(word.tuples)
  word.tuples$is_duplicate <- as.numeric(word.tuples$is_duplicate)
  setkeyv(word.tuples,c("sharedPhrases"))
  tuple.mean.duplicate <- word.tuples[,j=list(prob=mean(is_duplicate),count=length(is_duplicate)),by=list(sharedPhrases)]
  tuple.mean.duplicate  
}

# Tuple generation, words in both questions
nWordDiagnostics <- function(train, n=2, match=TRUE, stemming=FALSE, stopwords=FALSE) {
  if (match==TRUE) {
    word.tuples <- mapply(getNWordMatch, train$question1,train$question2,train$is_duplicate,n,stemming,USE.NAMES=FALSE, SIMPLIFY=TRUE)
  } else {
    word.tuples <- mapply(getNWordMiss, train$question1,train$question2,train$is_duplicate,n,stemming,stopwords,USE.NAMES=FALSE, SIMPLIFY=TRUE)
  }
  word.tuples <- list.clean(word.tuples, fun = is.null, recursive = FALSE)
  word.tuples <- list.rbind(word.tuples)
  word.tuples <- as.data.table(word.tuples)
  word.tuples$is_duplicate <- as.numeric(word.tuples$is_duplicate)
  setkeyv(word.tuples,c("sharedPhrases"))
  tuple.mean.duplicate <- word.tuples[,j=list(prob=mean(is_duplicate),count=length(is_duplicate)),by=list(sharedPhrases)]
  tuple.mean.duplicate  
}

WordMissProbTopN <- function(question1,question2,n=2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=FALSE, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
  }
  sharedPhrases1 <- NULL
  sharedPhrases2 <- NULL
  if (length(t1) >1 ) {
    sharedPhrases1 <- t(combn(t1,n))
    sharedPhrases1 <- t(apply(sharedPhrases1,1,function(x) x[order(x)]))
    sharedPhrases1 <- apply(sharedPhrases1,1,function(x) paste0(x,collapse=" "))
  }
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t2 <- wordStem(t2)
  }
  if (length(t2) >1) {
    sharedPhrases2 <- t(combn(t2,n))
    sharedPhrases2 <- t(apply(sharedPhrases2,1,function(x) x[order(x)]))
    sharedPhrases2 <- apply(sharedPhrases2,1,function(x) paste0(x,collapse=" "))
  }
  ret.prob <- NA
  ret.count <- NA
  if (length(sharedPhrases1>0) & length(sharedPhrases2>0)) {
    sharedPhrases <- union(setdiff(sharedPhrases1,sharedPhrases2),setdiff(sharedPhrases2,sharedPhrases1))
    if (length(sharedPhrases)>0) {
      matches <- nTupleTable[sharedPhrases]
      matches <- matches[!is.na(matches$count)]
      matches <- matches[order(prob,decreasing=decreasing)]
      ret.prob <-matches[matches$count > countMin]$prob[TopN]
      ret.count <-matches[matches$count > countMin]$count[TopN]
    } 
  }
  list(prob=ret.prob,count=ret.count)
}

WordMiss1ProbTopN <- function(question1,question2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=FALSE, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
  }
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t2 <- wordStem(t2)
  }
  sharedPhrases <- union(setdiff(t1,t2),setdiff(t2,t1))
  ret.prob <- NA
  ret.count <- NA
  if (length(sharedPhrases)>0) {
    matches <- nTupleTable[sharedPhrases]
    matches <- matches[!is.na(matches$count)]
    matches <- matches[order(prob,decreasing=decreasing)]
    ret.prob <-matches[matches$count > countMin]$prob[TopN]
    ret.count <-matches[matches$count > countMin]$count[TopN]
  } 
  list(prob=ret.prob,count=ret.count)
}

WordMatchProbTopN <- function(question1,question2,n=2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=TRUE, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
    t2 <- wordStem(t2)
  }
  sharedWords <- intersect(t1,t2)
  if (length(sharedWords) > (n-1)) {
    sharedPhrases <- t(combn(sharedWords,n))
    sharedPhrases <- t(apply(sharedPhrases,1,function(x) x[order(x)]))
    sharedPhrases <- apply(sharedPhrases,1,function(x) paste0(x,collapse=" "))
    matches <- nTupleTable[sharedPhrases]
    matches <- matches[!is.na(matches$count)]
    matches <- matches[order(prob,decreasing=decreasing)]
    #idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases & nTupleTable$count > countMin)
    ret.prob <-matches[matches$count > countMin]$prob[TopN]
    ret.count <-matches[matches$count > countMin]$count[TopN]
  } else {
    ret.prob <- NA 
    ret.count <- NA 
  }
  list(prob=ret.prob,count=ret.count)
}


# Tuple generation, phrases in both questions
nTupleDiagnostics <- function(n=2, train, tokenize=TRUE, match=TRUE, order=FALSE) {
  if (match==TRUE) {
    if (tokenize == TRUE) {
      word.tuples <- mapply(getNGramMatch, train$question1,train$question2,train$is_duplicate,n,USE.NAMES=FALSE, SIMPLIFY=TRUE)
    } else {
      word.tuples <- mapply(getNGramMatchPunct, train$question1,train$question2,train$is_duplicate,n,USE.NAMES=FALSE, SIMPLIFY=TRUE)
    }
  } else {
    word.tuples <-  mapply(getNGramMiss, train$question1,train$question2,train$is_duplicate,n,USE.NAMES=FALSE, SIMPLIFY=TRUE)
  }
  word.tuples <- list.clean(word.tuples, fun = is.null, recursive = FALSE)
  word.tuples <- list.rbind(word.tuples)
  word.tuples <- as.data.table(word.tuples)
  word.tuples$is_duplicate <- as.numeric(word.tuples$is_duplicate)
  setkeyv(word.tuples,c("sharedPhrases"))
  tuple.mean.duplicate <- word.tuples[,j=list(prob=mean(is_duplicate),count=length(is_duplicate)),by=list(sharedPhrases)]
  tuple.mean.duplicate  
}

TupleProbPunct <- function(question1,question2,n=2,nTupleTable=NULL) {
  t1 <- strsplit(tolower(question1), " ")
  t2 <- strsplit(tolower(question2), " ")
  t1 <- unlist(generate_ngrams_batch(t1,ngram_min =n,ngram_max=n))
  t2 <- unlist(generate_ngrams_batch(t2,ngram_min =n,ngram_max=n))
  sharedPhrases <- intersect(t1,t2)
  idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases)
  ret.prob <-sum(nTupleTable$prob[idx] * nTupleTable$count[idx])/sum(nTupleTable$count[idx])
  ret.count <-sum(nTupleTable$count[idx])
  list(prob=ret.prob,count=ret.count)
}

TupleProb <- function(question1,question2,n=2,nTupleTable=NULL) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- intersect(t1,t2)
  idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases)
  ret.prob <-sum(nTupleTable$prob[idx] * nTupleTable$count[idx])/sum(nTupleTable$count[idx])
  ret.count <-sum(nTupleTable$count[idx])
  list(prob=ret.prob,count=ret.count)
}


TupleProbTopN <- function(question1,question2,n=2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=TRUE) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- intersect(t1,t2)
  matches <- nTupleTable[sharedPhrases]
  matches <- matches[!is.na(matches$count)]
  matches <- matches[order(prob,decreasing=decreasing)]
  #idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases & nTupleTable$count > countMin)
  ret.prob <-matches[matches$count > countMin]$prob[TopN]
  ret.count <-matches[matches$count > countMin]$count[TopN]
  list(prob=ret.prob,count=ret.count)
}

TupleProbTopNMiss <- function(question1,question2,n=2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=TRUE) {
  t1 <- tokenize_ngrams(question1,n=n,n_min=n,simplify = TRUE)
  t2 <- tokenize_ngrams(question2,n=n,n_min=n,simplify = TRUE)
  sharedPhrases <- union(setdiff(t1,t2),setdiff(t2,t1))
  matches <- nTupleTable[sharedPhrases]
  matches <- matches[!is.na(matches$count)]
  matches <- matches[order(prob,decreasing=decreasing)]
  #idx <- which(nTupleTable$sharedPhrases %in% sharedPhrases & nTupleTable$count > countMin)
  ret.prob <-matches[matches$count > countMin]$prob[TopN]
  ret.count <-matches[matches$count > countMin]$count[TopN]
  list(prob=ret.prob,count=ret.count)
}

calcAndSaveNGramTables <- function(tupleSample, fileName = "Quora NGrams Train") {
  print("nTuple1")
  nTuple1 <- nTupleDiagnostics(1,train[tupleSample])
  print("nTuple1Miss")
  nTuple1Miss <- nTupleDiagnostics(1,train[tupleSample],tokenize=TRUE,match=FALSE)
  print("next")
  nTuple2 <- nTupleDiagnostics(2,train[tupleSample])
  print("next")
  nTuple2Punct <- nTupleDiagnostics(2,train[tupleSample],tokenize=FALSE)
  print("nTuple2Miss")
  nTuple2Miss <- nTupleDiagnostics(2,train[tupleSample],tokenize=TRUE,match=FALSE)
  print("nWordMatch2")
  nWordMatch2 <- nWordDiagnostics(train[tupleSample])
  print("next")
  nWordMatch2Stem <- nWordDiagnostics(train[tupleSample],match=TRUE,stemming=TRUE)
  print("next")
  nWordMatch3 <- nWordDiagnostics(train[tupleSample],n=3)
  print("next")
  nWordMatch3Stem <- nWordDiagnostics(train[tupleSample],n=3,match=TRUE,stemming=TRUE)
  print("next")
  nWordMiss2 <- nWordDiagnostics(train[tupleSample],2,match=FALSE)
  print("next")
  nWordMiss2Stem <- nWordDiagnostics(train[tupleSample],2,match=FALSE,stemming=TRUE)
  print("next")
  nCrossQuestion <- nCrossQuestionDiagnostics(train[tupleSample])
  print("next")
  nTuple3 <- nTupleDiagnostics(3,train[tupleSample])
  print("next")
  nTuple4 <- nTupleDiagnostics(4,train[tupleSample])
  print("next")
  nTuple5 <- nTupleDiagnostics(5,train[tupleSample])
  
  save("nTuple1","nTuple2","nTuple2Miss","nTuple3","nTuple4","nTuple5","nTuple1Miss","nTuple2Miss",
       "sample","tupleSample","nWordMatch2","nWordMatch2Stem","nWordMatch3Stem","nWordMatch3","nWordMiss2","nWordMiss2Stem",
       file=fileName)
}

traintestAddColumn <- function(sample,train) {
  j <- 1
  ptm <- proc.time()
  for (i in sample) {
    if (j %% 100 == 0) { 
      print(j)
      print(proc.time() - ptm)
    }
    train[i,nGramHitRate11_0 := nGramHitRate(question1,question2,1,1,0)]
    train[i,nGramHitRate22_0 := nGramHitRate(question1,question2,2,2,0)]
    
    # # # 
    t1 <- unlist(strsplit(tolower(train$question1[i]), " "))
    t2 <- unlist(strsplit(tolower(train$question2[i]), " "))
    #train[i,firstWordEqual := ifelse(t1[1]==t2[1],1,0)]
    # remove stop words
    t1 <- setdiff(t1,stopWords)
    t2 <- setdiff(t2,stopWords)
    sharedWords <- intersect(t1,t2)
    bothLengths <- length(t1)+length(t2)
    train[i,nGramHitRate := 2*length(sharedWords)/bothLengths]
    train[i,nGramHitRate := ifelse(is.infinite(nGramHitRate),0,nGramHitRate)]
    
    # Now N Tuples Max
    # ret <- TupleProbTopN(train$question1[i],train$question2[i],n=1,nTupleTable=nTuple1,countMin=0,TopN=1, decreasing=TRUE)
    # train[i,Tuple1ProbMax:= ret$prob]
    # train[i,Tuple1CountMax:= ret$count]
    # 
    ret <- TupleProbTopNMiss(train$question1[i],train$question2[i],n=1,nTupleTable=nTuple1Miss,countMin = 0, TopN = 1, decreasing=FALSE )
    train[i,Tuple1ProbTop1Min:= ret$prob]
    train[i,Tuple1CountTop1Min:= ret$count]
    
    ret <- TupleProbTopNMiss(train$question1[i],train$question2[i],n=2,nTupleTable=nTuple2Miss,countMin = 0, TopN = 1, decreasing=FALSE )
    train[i,Tuple2ProbTop1Min:= ret$prob]
    train[i,Tuple2CountTop1Min:= ret$count]
    
    ret <- WordMatchProbTopN(train$question1[i],train$question2[i],n=2,nTupleTable=nWordMatch2,countMin = 0, TopN = 1, decreasing=TRUE )
    train[i,WordMatch2ProbTop1:= ret$prob]
    train[i,WordMatch2CountTop1:= ret$count]
    
    # ret <- WordMatchProbTopN(train$question1[i],train$question2[i],n=3,nTupleTable=nWordMatch3,countMin = 0, TopN = 1, decreasing=TRUE )
    # train[i,WordMatch3ProbTop1:= ret$prob]
    # train[i,WordMatch3CountTop1:= ret$count]
    # 
    ret <- WordMissProbTopN(train$question1[i],train$question2[i],n=2,nTupleTable=nWordMiss2,countMin = 0, TopN = 1, decreasing=FALSE )
    train[i,WordMiss2ProbTop1:= ret$prob]
    train[i,WordMiss2CountTop1:= ret$count]
    
    ret <- WordMissProbTopN(train$question1[i],train$question2[i],n=2,nTupleTable=nWordMiss2,countMin = 0, TopN = 1, decreasing=TRUE )
    train[i,WordMiss2ProbTop1Max:= ret$prob]
    train[i,WordMiss2CountTop1Max:= ret$count]
    
    # ret <- WordMissProbTopN(train$question1[i],train$question2[i],n=2,nTupleTable=nWordMiss2,countMin = 0, TopN = 2, decreasing=FALSE )
    # train[i,WordMiss2ProbTop2:= ret$prob]
    # train[i,WordMiss2CountTop2:= ret$count]
    # 
    # ret <- WordMatchProbTopN(train$question1[i],train$question2[i],n=2,
    #                          nTupleTable=nWordMatch2Stem,countMin = 0, TopN = 1, decreasing=TRUE, stemming=TRUE )
    # train[i,WordMatch2ProbTop1Stem:= ret$prob]
    # train[i,WordMatch2CountTop1Stem:= ret$count]
    # 
    j = j+1
  }

  train
}

WordMissProbTopNCheck <- function(question1,question2,n=2,nTupleTable=NULL,TopN=1, decreasing=FALSE, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
  }
  sharedPhrases1 <- NULL
  sharedPhrases2 <- NULL
  if (length(t1) >1 ) {
    sharedPhrases1 <- t(combn(t1,n))
    sharedPhrases1 <- t(apply(sharedPhrases1,1,function(x) x[order(x)]))
    sharedPhrases1 <- apply(sharedPhrases1,1,function(x) paste0(x,collapse=" "))
  }
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t2 <- wordStem(t2)
  }
  if (length(t2) >1) {
    sharedPhrases2 <- t(combn(t2,n))
    sharedPhrases2 <- t(apply(sharedPhrases2,1,function(x) x[order(x)]))
    sharedPhrases2 <- apply(sharedPhrases2,1,function(x) paste0(x,collapse=" "))
  }
  ret.prob <- NA
  ret.count <- NA
  if (length(sharedPhrases1>0) & length(sharedPhrases2>0)) {
    sharedPhrases <- union(setdiff(sharedPhrases1,sharedPhrases2),setdiff(sharedPhrases2,sharedPhrases1))
    if (length(sharedPhrases)>0) {
      matches <- nTupleTable[sharedPhrases]
      matches <- matches[!is.na(matches$count)]
      matches <- matches[order(prob,decreasing=decreasing)]
      upper <- min(TopN,nrow(matches))
      #ret.prob <-sum(matches$prob[1:upper]*matches$count[1:upper])/sum(matches$count[1:upper])
      ret.prob <-mean(matches$prob[1:upper])
      ret.count <-sum(matches$count[1:upper])
    } 
  }
  list(prob=ret.prob,count=ret.count)
}

WordMissProbAllTopN <- function(question1,question2,n=2,nTupleTable=NULL,countMin=0,TopN=1, decreasing=FALSE, stemming=FALSE) {
  t1 <- tokenize_ngrams(question1,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t1 <- wordStem(t1)
  }
  sharedPhrases1 <- NULL
  sharedPhrases2 <- NULL
  if (length(t1) >1 ) {
    sharedPhrases1 <- t(combn(t1,n))
    sharedPhrases1 <- t(apply(sharedPhrases1,1,function(x) x[order(x)]))
    sharedPhrases1 <- apply(sharedPhrases1,1,function(x) paste0(x,collapse=" "))
  }
  t2 <- tokenize_ngrams(question2,n=1,n_min=1,simplify = TRUE)
  if (stemming==TRUE) {
    t2 <- wordStem(t2)
  }
  if (length(t2) >1) {
    sharedPhrases2 <- t(combn(t2,n))
    sharedPhrases2 <- t(apply(sharedPhrases2,1,function(x) x[order(x)]))
    sharedPhrases2 <- apply(sharedPhrases2,1,function(x) paste0(x,collapse=" "))
  }
  ret.prob <- NA
  ret.count <- NA
  if (length(sharedPhrases1>0) & length(sharedPhrases2>0)) {
    sharedPhrases <- union(setdiff(sharedPhrases1,sharedPhrases2),setdiff(sharedPhrases2,sharedPhrases1))
    if (length(sharedPhrases)>0) {
      matches <- nTupleTable[sharedPhrases]
      matches <- matches[!is.na(matches$count)]
      matches <- matches[order(prob,decreasing=decreasing)]
      ret.prob <-matches[matches$count > countMin]$prob[1:TopN]
      ret.count <-matches[matches$count > countMin]$count[1:TopN]
    } 
  }
  list(prob=ret.prob,count=ret.count)
}
