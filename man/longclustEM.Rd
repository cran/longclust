\name{longclustEM}
\alias{longclustEM}
\title{Model-Based Clustering and Classification for Longitudinal Data}
\description{
Carries out model-based clustering or classification using multivariate t or Gaussian mixture 
models with Cholesky decomposed covariance structure. EM algorithms are used for parameter 
estimation and the BIC is used for model selection.
}
\usage{
longclustEM(x, Gmin, Gmax, class=NULL, linearMeans = FALSE, 
modelSubset = NULL, initWithKMeans = FALSE, criteria = "BIC", 
equalDF = FALSE, gaussian=FALSE,  userseed=1004)
}
\arguments{
\item{x}{A matrix or data frame such that rows correspond to observations and columns correspond to 
variables.}
\item{Gmin}{A number giving the minimum number of components to be used.}
\item{Gmax}{A number giving the maximum number of components to be used.}
\item{class}{If \code{NULL} then model-based clustering is performed. If a vector with length equal 
to the number of observations, then model-based classification is performed. In this latter case, 
the ith entry of \code{class} is either zero, indicating that the component membership of 
observation i is unknown, or it corresponds to the component membership of observation i.}
\item{linearMeans}{If TRUE, then means are modelled using linear models.}
\item{modelSubset}{A vector of strings giving the models to be used. If set to NULL, all models are 
used.}
\item{initWithKMeans}{If TRUE, the components are initialized using k-means algorithm.}
\item{criteria}{A string that denotes the criteria used for evaluating the models. Its value should 
be "BIC" or "ICL".}
\item{equalDF}{If TRUE, the degrees of freedom of all the components will be the same.}
\item{gaussian}{If TRUE, a mixture of Gaussian distributions is used in place of a mixture of 
t-distributions.}
\item{userseed}{The random number seed to be used.}
}
\value{
\item{Gbest}{The number of components for the best model.}
\item{zbest}{A matrix that gives the probabilities for any data element to belong to any component 
in the best model.}
\item{nubest}{ A vector of \code{Gbest} integers, that give the degrees of freedom for each 
component in the best model.}
\item{mubest}{ A matrix containing the means of the components for the best model (one per row).}
\item{Tbest}{ A list of \code{Gbest} matrices, giving the T matrices of the components for the best 
model.}
\item{Dbest}{ A list of \code{Gbest} matrices, giving the D matrices of the components for the best 
model.}
}
\references{
Paul D. McNicholas and T. Brendan Murphy (2010). Model-based clustering of longitudinal data. 
\emph{The Canadian Journal of Statistics} \bold{38}(1), 153-168.

Paul D. McNicholas and Sanjeena Subedi (2012). Clustering gene expression time course data using 
mixtures of multivariate t-distributions. \emph{Journal of Statistical Planning and Inference} 
\bold{142}(5), 1114-1127.
}
\author{
Paul D. McNicholas, K. Raju Jampani and Sanjeena Subedi
}
\examples{
library(mvtnorm)
m1 <- c(23,34,39,45,51,56)
S1 <- matrix(c(1.00, -0.90, 0.18, -0.13, 0.10, -0.05, -0.90, 
1.31, -0.26, 0.18, -0.15, 0.07, 0.18, -0.26, 4.05, -2.84, 
2.27, -1.13, -0.13, 0.18, -2.84, 2.29, -1.83, 0.91, 0.10, 
-0.15, 2.27, -1.83, 3.46, -1.73, -0.05, 0.07, -1.13, 0.91, 
-1.73, 1.57), 6, 6)
m2 <- c(16,18,15,17,21,17)
S2 <- matrix(c(1.00, 0.00, -0.50, -0.20, -0.20, 0.19, 0.00, 
2.00, 0.00, -1.20, -0.80, -0.36,-0.50, 0.00, 1.25, 0.10, 
-0.10, -0.39, -0.20, -1.20, 0.10, 2.76, 0.52, -1.22,-0.20, 
-0.80, -0.10, 0.52, 1.40, 0.17, 0.19, -0.36, -0.39, -1.22, 
0.17, 3.17), 6, 6)
m3 <- c(8, 11, 16, 22, 25, 28)
S3 <- matrix(c(1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
1.00, -0.20, -0.64, 0.26, 0.00, 0.00, -0.20, 1.04, -0.17, 
-0.10, 0.00, 0.00, -0.64, -0.17, 1.50, -0.65, 0.00, 0.00, 
0.26, -0.10, -0.65, 1.32, 0.00, 0.00, 0.00, 0.00, 0.00, 
0.00, 1.00), 6, 6)
m4 <- c(12, 9, 8, 5, 4 ,2)
S4 <- diag(c(1,1,1,1,1,1))
data <- matrix(0, 40, 6)
data[1:10,] <- rmvnorm(10, m1, S1)
data[11:20,] <- rmvnorm(10, m2, S2)
data[21:30,] <- rmvnorm(10, m3, S3)
data[31:40,] <- rmvnorm(10, m4, S4)
clus <- longclustEM(data, 3, 5, linearMeans=TRUE)
summary(clus)
plot(clus,data)
}
\keyword{"multivariate"}
\keyword{"cluster"}
\keyword{"classif"}
