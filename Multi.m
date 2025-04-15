clc;
close all;
clear;

%% Parameters
r = 0.033;    % Wheel radius (m)
l = 0.2;      % Axle length (m)
D = 5;        % Noise coefficient
v = 0.5;      % Velocity
d = 3;        % Dimensional parameter

%% Initial Positions (x, y, theta)
x_1 = [0; 0; 0];
x_2 = [-1; 1; 0];
x_3 = [-1; -1; 0];

%% Compute SE(2) Transformations
a_1 = compute_SE2(x_1);
a_2 = compute_SE2(x_2);
a_3 = compute_SE2(x_3);

%% Define Lie Algebra Basis for SE(2)
E = {
    [0 0 1; 0 0 0; 0 0 0],  % E1
    [0 0 0; 0 0 1; 0 0 0],  % E2
    [0 -1 0; 1 0 0; 0 0 0]  % E3
};

sim_time = 1.5;
dt = 0.01;

%% Control Inputs
w1 = v / r;       % Left wheel angular velocity (rad/s)
w2 = v / r;       % Right wheel angular velocity (rad/s)
omega = (w1 + w2) / 2;

%% Covariance Matrices of the measurement
Sigma_m = [0.01  0.02   0.001;
                  0.02  0.25   0.015;
                  0.001 0.015 0.15];
Sigma_12 = Sigma_m;
Sigma_13 = Sigma_m;

num_samples = 1000; % Number of samples for visualization

%% Sample Trajectories
samples = zeros(num_samples, 3, 3, 3);

for n = 1:num_samples
    g_1 = a_1;
    g_2 = a_2;
    g_3 = a_3;
    
    % Propagate dynamics
    for j = 1:sim_time/dt
        g_1 = Dynamics(dt, g_1, r, l, D, w1, w2);
        g_2 = Dynamics(dt, g_2, r, l, D, w1, w2);
        g_3 = Dynamics(dt, g_3, r, l, D, w1, w2);
    end
    
    % Store samples
    samples(n,1,:,:) = g_1;
    samples(n,2,:,:) = g_2;
    samples(n,3,:,:) = g_3;
end

%% Prediction (propagation), relative pose
[mu_pred_1, Sigma_pred_1] = mu_Sigma_prediction(r, D, l, omega, sim_time);
[mu_pred_2, Sigma_pred_2] = mu_Sigma_prediction(r, D, l, omega, sim_time);
[mu_pred_3, Sigma_pred_3] = mu_Sigma_prediction(r, D, l, omega, sim_time);

%% modeling the sensing process (with noise); use the last sample path as the ground truth
[m_12, m_13] = compute_relative_pose(g_1, g_2, g_3);
m_12 = measurement_model(m_12, Sigma_12);
m_13 = measurement_model(m_13, Sigma_13);

%% update the pose of robot1 based on the sensor measurements m_12, m_13
[Sigma_update_1, mu_update_1] = mu_Sigma_update(m_12, m_13, Sigma_12, Sigma_13, d, mu_pred_1, Sigma_pred_1, E, mu_pred_2, mu_pred_3, a_1, a_2, a_3 );

%% Plots
%% Plotting the robot samples and predicted/true poses
ax = gca();            % Get current axes for plotting
hold(ax, 'on');        % Hold the current plot to overlay multiple scatter plots

% Plot samples for Robot 1, Robot 2, and Robot 3
scatter(samples(:,1,1,3), samples(:,1,2,3), '.', 'DisplayName', 'Robot 1 Samples');
scatter(samples(:,2,1,3), samples(:,2,2,3), '.', 'DisplayName', 'Robot 2 Samples');
scatter(samples(:,3,1,3), samples(:,3,2,3), '.', 'DisplayName', 'Robot 3 Samples');

% Plot the true poses for the robots
scatter([g_1(1,3); g_2(1,3); g_3(1,3)], [g_1(2,3), g_2(2,3), g_3(2,3)], '*', 'DisplayName', 'True Poses');

% Plot the predicted pose for Robot 1
scatter(mu_update_1(1,3), mu_update_1(2,3), '^','k', 'DisplayName', 'Predicted Pose for Robot 1');

% Add labels and legend for clarity
xlabel('X Position (m)');
ylabel('Y Position (m)');
legend('show');      % Show the legend to identify each plot
grid on;             % Turn on the grid for better visualization
title('Robot Sample and Pose Visualization');


function [Sigma_update, mu_update] = mu_Sigma_update(m_12, m_13, Sigma_12, Sigma_13, d, mu_pred, Sigma_pred, E, mu_pred_2, mu_pred_3, a_1, a_2, a_3)

    % Compute covariance updates
    A_12 = compute_covariance_update(m_12, Sigma_pred);
    B_12 = Sigma_12;
    
    A_13 = compute_covariance_update(m_13^-1, Sigma_pred);
    B_13 = Sigma_13;

    % Compute F(A, B) terms
    F_AB_12 = compute_F_AB(A_12, B_12, E, d);
    F_AB_13 = compute_F_AB(A_13, B_13, E, d);

    % Compute updated covariance matrices
    Sigma_212 = A_12 + B_12 + F_AB_12;
    Sigma_313 = A_13 + B_13 + F_AB_13;

    % Estimate state of robot 1 using measurements m12 and m13
    q_2 = m_12 * inv(mu_pred_2) * inv(a_2) * a_1 * mu_pred;
    x_hat_2 = logm(q_2);
    Gamma_2 = eye(size(x_hat_2)) + 0.5 * adj_s(x_hat_2);
    S_2 = Gamma_2' * inv(Adj(m_12))' * inv(Sigma_212) * inv(Adj(m_12)) * Gamma_2;

    q_3 = m_13 * inv(mu_pred_3) * inv(a_3) * a_1 * mu_pred;
    x_hat_3 = logm(q_3);
    Gamma_3 = eye(size(x_hat_3)) + 0.5 * adj_s(x_hat_3);
    S_3 = Gamma_3' * inv(Adj(m_13))' * inv(Sigma_313) * inv(Adj(m_13)) * Gamma_3;

    % Compute S_1 and the final estimates
    S_1 = inv(Sigma_pred);
    S_bar_prime = S_1 + S_2 + S_3;
    x_bar_hat_prime = inv(S_bar_prime) * (S_2 * vee(x_hat_2) + S_3 * vee(x_hat_3));

    % Compute final updated covariance and mean
    Gamma_bar_prime = Gamma_2 + Gamma_3 + eye(3); % Identity matrix for Gamma_1
    Sigma_update = Gamma_bar_prime * inv(S_bar_prime) * Gamma_bar_prime';
    mu_update = a_1 * mu_pred * expm(-hat(x_bar_hat_prime));

end

%% Helper Functions
function A = compute_covariance_update(m, Sigma_pred)
    % Compute the covariance update A = Adj(m^-1) * Sigma_pred * Adj(m^-1)'
    A = Adj(inv(m)) * Sigma_pred * Adj(inv(m))';
end

function F_AB = compute_F_AB(A, B, E, d)
    % Compute F(A, B) using the ad_s operator
    F_AB = 0;
    for i = 1:d
        for j = 1:d
            A_double_prime = adj_s(E{i}) * adj_s(E{j}) * A(i,j);
            B_double_prime = adj_s(E{i}) * adj_s(E{j}) * B(i,j);

            F_AB = F_AB + (1/4) * (adj_s(E{i}) * B * adj_s(E{j})' * A(i,j)) + ...
                          (1/12) * (A_double_prime * B + B' * A_double_prime') + ...
                          (1/12) * (B_double_prime * A + A' * B_double_prime');
        end
    end
end


%% JK: Sampling SDEs on Lie Groups  
% Reference: https://arxiv.org/pdf/2401.03425, Equation (13)  
function   g = Dynamics( dt, g, r, l, D, w1, w2)
% Current orientation

        % Generate Wiener increments for noise
        d_omega1 = sqrt(dt) * randn; % Noise for wheel 1
        d_omega2 = sqrt(dt) * randn; % Noise for wheel 2
        d_omega = [d_omega1; d_omega2];

        % Update the state using kinematics and noise
        dx = [r/2 * (w1 + w2); ...
                 0; ...
                 r / 2 * (w1 - w2)] * dt ...
             + sqrt(D) * [r/2 , r/2; ...
                              0, 0; ...
                              r / l, -r / l] * d_omega;    
g = g * expm(hat(dx));

end


function g = compute_SE2(x)
    % Computes the SE(2) transformation matrix for a given pose [x; y; theta]
    theta = x(3);
    g = [cos(theta), -sin(theta), x(1);
         sin(theta),  cos(theta), x(2);
         0,          0,          1];
end

function mat = hat(v)
  
    mat = zeros(3);
    
    % Assign values based on the hat operator definition
    mat(1,3) = v(1);  % Translation in x
    mat(2,3) = v(2);  % Translation in y
    mat(2,1) = v(3);  % Rotation component
    mat(1,2) = -v(3); % Skew-symmetric property
end

function [mu_pred, Sigma_pred] = mu_Sigma_prediction(r, D, l, omega, t)
    % Computes the mean and covariance propagation for SE(2) system dynamics.
   
    % Compute mean propagation
    mu_pred = [1, 0, r * omega * t;
               0, 1, 0;
               0, 0, 1];

    % Compute covariance evolution analytically
    Sigma_pred = [0.5 * D * r^2 * t, 0, 0;
                  0, (2 * D * omega^2 * r^4 * t^3) / (3 * l^2), (D * omega * r^3 * t^2) / l^2;
                  0, (D * omega * r^3 * t^2) / l^2, (2 * D * r^2 * t) / l^2];
end

function [m_12, m_13] = compute_relative_pose (g_1, g_2, g_3)
m_12 = g_1^-1 * g_2;
m_13 = g_1^-1 * g_3;

end

function [mat] = measurement_model(mean, cov)
    % Measurement model for generating a state with added noise based on
    
     L = chol(cov, 'lower');

    % Generate a random noise vector and compute the measurement model
    mat = mean * expm(hat(L.' * randn(3, 1)));
end

function [matrix2] = Adj(matrix1)
    
  
    matrix2 = zeros(3);
    
   
    matrix2(1:2, 1:2) = matrix1(1:2, 1:2);
    
    % Compute the skew-symmetric part for the translational components
    matrix2(1, 3) = matrix1(2, 3);
    matrix2(2, 3) = -matrix1(1, 3);
    
    % The last element is fixed as 1
    matrix2(3, 3) = 1;
end

function [mat2] = adj_s(mat1)
    % Computes the adjoint matrix for an SE(2) matrix.
    
    
    mat2 = zeros(3);
    
    mat2(1:2, 1:2) = mat1(1:2, 1:2);
    
    % Compute the skew-symmetric part for the translational components
    mat2(1, 3) = mat1(2, 3);
    mat2(2, 3) = -mat1(1, 3);
end

function [v] = vee(mat1)
      
    % Extract the elements from the skew-symmetric matrix
    v = [mat1(1, 3);   
         mat1(2, 3);   
         mat1(2, 1)];  
end
