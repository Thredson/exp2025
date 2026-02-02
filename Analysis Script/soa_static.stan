// Static Subjective Ordering Analysis (SOA) Model

data {
  int<lower=1> N;                    // Number of trials
  int<lower=1> K;                    // Number of stimuli (6 for ABCDEF)
  int<lower=1, upper=K> left[N];     // Left stimulus in each trial
  int<lower=1, upper=K> right[N];    // Right stimulus in each trial
  int<lower=0, upper=1> choice[N];   // 0 = chose left, 1 = chose right
  int<lower=0, upper=1> reward[N];   // 0 = no reward, 1 = reward
  int<lower=0, upper=1> is_training[N]; // 1 if training phase, 0 if test
}

parameters {
  // Position parameters for each stimulus on the linear continuum
  vector[K] z_score;                // Z-scores (positions) for each stimulus
  
  // Uncertainty (standard deviation) for each stimulus
  vector<lower=0>[K] sigma;          // Standard deviations for each stimulus
  
  // Decision parameters
  real<lower=0, upper=1> theta;      // theta - probability of random choice
}

transformed parameters {
  // Track evolving estimates if needed for diagnostic purposes
  vector[N] log_lik;                 // Log likelihood for each trial
  
  // Calculate choice probabilities for each trial
  for (n in 1:N) {
    real diff_mean = z_score[right[n]] - z_score[left[n]];
    real combined_sigma = sqrt(sigma[left[n]]^2 + sigma[right[n]]^2);
    
    // Probability of choosing right stimulus (using probit-like decision rule)
    real p_right_deterministic = Phi_approx(diff_mean / (combined_sigma));
    
    // Account for theta
    real p_right = theta * 0.5 + (1 - theta) * p_right_deterministic;
    
    // Calculate log likelihood
    if (choice[n] == 1) {
      log_lik[n] = log(p_right);
    } else {
      log_lik[n] = log(1 - p_right);
    }
  }
}

model {
  // Priors on stimulus positions

  z_score ~ normal(0, 1);            // Standard normal prior on positions
  
  // Priors on uncertainty parameters
  sigma ~ exponential(1);            // Exponential prior on standard deviations
  
  // Priors on decision parameters
  theta ~ beta(1, 1);                // Beta prior for theta
  
  // Likelihood
  target += sum(log_lik);
}

generated quantities {
  // Posterior predictive checks
  int y_pred[N];                     // Predicted choices
  
  // Generate predictions for each trial
  for (n in 1:N) {
    real diff_mean = z_score[right[n]] - z_score[left[n]];
    real combined_sigma = sqrt(sigma[left[n]]^2 + sigma[right[n]]^2);
    real p_right_deterministic = Phi_approx(diff_mean / (combined_sigma));
    real p_right = theta * 0.5 + (1 - theta) * p_right_deterministic;
    
    y_pred[n] = bernoulli_rng(p_right);
  }
  
  // Calculate subjective ordering strength (gradient of z-scores)
  real ordering_strength = max(z_score) - min(z_score);
  
  // Calculate average uncertainty
  real mean_uncertainty = mean(sigma);
}


