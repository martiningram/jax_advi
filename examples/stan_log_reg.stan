data {
  int N; // Num data points
  int K; // Number of coefficients
    
  matrix[N, K] X; // design matrix
  int y[N]; // Outcomes
}
parameters {
  vector[K] beta;
  real gamma;
}
model {
  beta ~ normal(0, 1);
  gamma ~ normal(0, 1);
  y ~ bernoulli_logit(X * beta + gamma);
}
