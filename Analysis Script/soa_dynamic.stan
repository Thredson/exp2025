// Dynamic Subjective Ordering Analysis (SOA) Model

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
  // Initial position parameters
  vector[K] z_score_init;           // Initial z-scores for each stimulus
  
  // Initial uncertainty
  vector<lower=0>[K] sigma_init;     // Initial standard deviations
  
  // Decision parameters
  real<lower=0, upper=1> theta;      // random choice probability
  
  // Learning parameters
  real<lower=0, upper=1> alpha_pos;  // Learning rate for position updates
  real<lower=0, upper=1> alpha_neg;  // Learning rate for negative feedback
  real<lower=0, upper=1> sigma_decay; // Rate of uncertainty reduction
}

transformed parameters {
  // Dynamic tracking of positions and uncertainties
  matrix[N+1, K] z_score_trajectory;  // Position trajectory
  matrix[N+1, K] sigma_trajectory;    // Uncertainty trajectory
  vector[N] log_lik;                  // Log likelihood
  
  // Initialize trajectories
  z_score_trajectory[1, :] = z_score_init';
  sigma_trajectory[1, :] = sigma_init';
  
  // Update positions and uncertainties trial by trial
  for (n in 1:N) {
    int chosen_stim;
    int unchosen_stim;
    real update_magnitude;
    
    // Determine which stimulus was chosen
    if (choice[n] == 0) {
      chosen_stim = left[n];
      unchosen_stim = right[n];
    } else {
      chosen_stim = right[n];
      unchosen_stim = left[n];
    }
    
    // Calculate choice probability for this trial
    real diff_mean = z_score_trajectory[n, right[n]] - z_score_trajectory[n, left[n]];
    real combined_sigma = sqrt(sigma_trajectory[n, left[n]]^2 + 
                               sigma_trajectory[n, right[n]]^2);
    real p_right_det = Phi_approx(diff_mean / combined_sigma);
    real p_right = theta * 0.5 + (1 - theta) * p_right_det;
    
    // Log likelihood
    if (choice[n] == 1) {
      log_lik[n] = log(p_right);
    } else {
      log_lik[n] = log(1 - p_right);
    }
    
    // Update positions based on feedback
    z_score_trajectory[n+1, :] = z_score_trajectory[n, :];
    sigma_trajectory[n+1, :] = sigma_trajectory[n, :];
    
    if (is_training[n] == 1) {
      // Only update during training phase
      if (reward[n] == 1) {
        // Positive feedback: consolidate current positions
        sigma_trajectory[n+1, chosen_stim] = sigma_trajectory[n, chosen_stim] * (1 - sigma_decay);
        sigma_trajectory[n+1, unchosen_stim] = sigma_trajectory[n, unchosen_stim] * (1 - sigma_decay);
      } else {
        // Negative feedback: adjust positions
        update_magnitude = alpha_neg * (sigma_trajectory[n, chosen_stim] + 
                                        sigma_trajectory[n, unchosen_stim]);
        
        // Move chosen stimulus down, unchosen stimulus up
        z_score_trajectory[n+1, chosen_stim] = z_score_trajectory[n, chosen_stim] - 
                                               update_magnitude / 2;
        z_score_trajectory[n+1, unchosen_stim] = z_score_trajectory[n, unchosen_stim] + 
                                                 update_magnitude / 2;
        
        // Slightly increase uncertainty after error
        sigma_trajectory[n+1, chosen_stim] = sigma_trajectory[n, chosen_stim] * 
                                             (1 + sigma_decay * 0.5);
        sigma_trajectory[n+1, unchosen_stim] = sigma_trajectory[n, unchosen_stim] * 
                                               (1 + sigma_decay * 0.5);
      }
    }
  }
}

model {
  // Priors on initial positions
  z_score_init ~ normal(0, 1);
  
  // Priors on initial uncertainties
  sigma_init ~ exponential(1);
  
  // Priors on decision parameters
  theta ~ beta(1, 1);
  
  // Priors on learning parameters
  alpha_pos ~ beta(2, 5);             // Favor smaller positive learning
  alpha_neg ~ beta(2, 2);             // Symmetric prior for negative learning
  sigma_decay ~ beta(2, 5);           // Favor gradual decay
  
  // Likelihood
  target += sum(log_lik);
}

generated quantities {
  // Final estimated positions and uncertainties
  vector[K] z_score_final = z_score_trajectory[N+1, :]';
  vector[K] sigma_final = sigma_trajectory[N+1, :]';
  
  // Ordering strength at end of training
  real ordering_strength_final = max(z_score_final) - min(z_score_final);
  
  // Predict choices for test phase
  int y_pred[N];
  for (n in 1:N) {
    real diff_mean = z_score_trajectory[n, right[n]] - z_score_trajectory[n, left[n]];
    real combined_sigma = sqrt(sigma_trajectory[n, left[n]]^2 + 
                               sigma_trajectory[n, right[n]]^2);
    real p_right_det = Phi_approx(diff_mean / combined_sigma);
    real p_right = theta * 0.5 + (1 - theta) * p_right_det;
    y_pred[n] = bernoulli_rng(p_right);
  }
}


