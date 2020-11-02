data {
  int N; // number of matches
  int P; // number of players

  int winner_ids[N];
  int loser_ids[N];
}
parameters {
  real<lower=0> prior_sd;
  vector[P] player_skills_raw;
}
transformed parameters {
  vector[P] player_skills;
  player_skills = player_skills_raw * prior_sd;
}
model {
  player_skills_raw ~ normal(0, 1);
  prior_sd ~ normal(0, 1);

  rep_array(1, N) ~ bernoulli_logit(player_skills[winner_ids] - player_skills[loser_ids]);
}
