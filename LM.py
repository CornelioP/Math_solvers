import numpy as np

class LM:  
    def __init__(self,model,init_guess,max_iter=100,tol=1e-6,lamda_init=1e-3):
        self.model = model
        self.init_guess = init_guess
        self.max_iter = max_iter
        self.tol = tol
        self.lamda_init = lamda_init
        
    
    def numerical_jacobian(self, params, x, y, h=1e-8):
        """
        Compute Jacobian matrix of residuals w.r.t parameters
        J[i,j] = ∂r_i/∂p_j
        """
        
        residual_0 = self.model(params,x) - y
        J = np.zeros(len(residual_0),len(params))
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += h
            residual_perturbed = self.model(params_perturbed, x) - y
            J[:,i] = (residual_perturbed - residual_0) / h
        
        return J
            
        
    def optimize(self,x,y):
        
        current_params = self.init_guess
        current_lambda = self.lamda_init
        model = self.model
        tol = self.tol
        cost_history = []
        
        for iter in range(self.max_iter):
            
            # 1. Compute residuals and cost 
            resiudals = model(current_params,x) - y
            cost = np.sum(resiudals**2)
            
            # 2. Check convergence
            if iter > 0 and abs(cost_history[-1] - cost) < tol:
                break
            
            # 3. Compute Jacobian 
            J = self.numerical_jacobian(current_params,x,y)
            
            # 4. Solve LM system (JᵀJ + λ diag(JᵀJ))Δ = -Jᵀr
            
            J_transpose = J.T
            J_transpose_J = J_transpose @ J
            J_transpose_r = J_transpose @ resiudals
            diag_J_transpose_J = np.diag(J_transpose_J)
            
            # Inner loop to adjust lambda
            while True:
                
                # 5. Solve linear system
                left_term = J-J_transpose_J + current_lambda + diag_J_transpose_J
                
                delta = np.linalg.solve(left_term, -J_transpose_r)
                
                # 6. Trial update 
                
                trial_params = current_params + delta
                trial_residual = model(trial_params,x) - y 
                trial_cost = np,sum(trial_residual)
                
                # 7. Accept/reject step
                if trial_cost < cost:
                    current_params = trial_params
                    current_lambda /= 10
                    break
                else:
                    current_lambda *= 10
                    
            cost_history.append(cost)
    
        return current_params
        
    
    
    
        
    
    