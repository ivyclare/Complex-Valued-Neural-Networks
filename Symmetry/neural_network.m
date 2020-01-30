function [wIH, wHO, O] = neural_network (X, T)
% INITIALIZE VALUES
sizeI   = 32 +1;            % number of input neurons
sizeO   = 32;               % number of output neurons
sizeH   =  5 +1;            % number of hidden neurons
k       = 0.1;              % learning constant
n       = 2;     % number of samples
O  = zeros(n, sizeO);  % matrix to save the output
Ht = zeros;

iteration = 3001;   % Number of iterations

% INITIALIZING weights with random values
wIH = rand(sizeH, sizeI);   % weights from input layer to hidden layer
wHO = rand(sizeO, sizeH);   % weights from hidden layer to output layer

deltaEwHI1 = zeros(sizeH, sizeI);
deltaEwHI2 = zeros(sizeH, sizeI);

deltaEwOH1 = zeros(sizeO, sizeH);
deltaEwOH2 = zeros(sizeO, sizeH);

% Counter to get error matrix
counter = 1;
er_matrix = zeros();

while counter < iteration
    er = 0;
    for r = 1:n
        % So that all the inputs are at a comparable range, we normalize     
        % normalizing Input
        if sum(X(r, :)) > 1
            X(r, :) = X(r, :)/ 1000;
        end
        
        % normalizing Output
        if sum(T(r, :)) > 1
            T(r, :) = T(r, :) / 1000;
        end
        
        % calculate the hidden layer output H
        wXH    = (wIH * X(r, :).').';    
        % apply activation function tanh(x)
        H      = tanh(abs(wXH)) .* exp(1i * angle(wXH));

        % calculate the final output 
        wOH    = (wHO * H.').';                           
        zO      = tanh(abs(wOH)).* exp(1i * angle(wOH)); 
        
        
        % calculating the error value. Compare desired output to derived
        % output
        temp    = abs((zO - T(r, :))).^2;
        er      = (1/2) .* sum( temp ) ;
        
        zOt = T(r, :);
        zI  = X(r, :);
        % Updating Weights
        for ii = 1:sizeO
          for jj = 1:sizeH           
            deltaEwOH1(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zOt(ii))) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii))) ).' * abs(H(jj)) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii)) - angle(wHO(ii,jj))).' - ...
                     abs(zO(ii)) * abs(zOt(ii)) * sin(angle(zO(ii)) - angle(zOt(ii))) .* ...
                     (abs(H(jj)) / atanh(abs(zO(ii)))).* ...
                     sin(angle(zO(ii)) - angle(zOt(ii) - angle(wHO(ii,jj))));
          end
        end
        
        % A few changes with this update
        for ii = 1:sizeO
          for jj = 1:sizeH           
            deltaEwOH2(ii,jj) =  (1- abs(zO(ii)).^2).' .* ( (abs(zO(ii)) - abs(zOt(ii))) .* ...
                     cos(angle(zO(ii)) - angle(zOt(ii))) ).' * abs(H(jj)) .* ...
                     sin(angle(zO(ii)) - angle(zOt(ii)) - angle(wHO(ii,jj))).' + ...
                     abs(zO(ii)) * abs(zOt(ii)) * sin(angle(zO(ii)) - angle(zOt(ii))) .* ...
                     (abs(H(jj)) / atanh(abs(zO(ii)))).* ...
                     cos(angle(zO(ii)) - angle(zOt(ii) - angle(wHO(ii,jj))));

          end
        end
        
                
        % the same as above, but the indexes o and h are
        % replaced with h and i
        
        for h = 1:sizeH
            Ht(h) = (tanh(zOt * wHO(:,h)).' * exp(1i * angle(zOt * wHO(:,h)))).';
        end
        
        
        for ii = 1:sizeH
          for jj = 1:sizeI           
            deltaEwHI1(ii,jj) =  (1- abs(H(ii)).^2).' .* ( (abs(H(ii)) - abs(Ht(ii))) .* ...
                     cos(angle(H(ii)) - angle(Ht(ii))) ).' * abs(zI(jj)) .* ...
                     cos(angle(H(ii)) - angle(Ht(ii)) - angle(wIH(ii,jj))).' - ...
                     abs(H(ii)) * abs(Ht(ii)) * sin(angle(H(ii)) - angle(Ht(ii))) .* ...
                     (abs(zI(jj)) / atanh(abs(H(ii)))).* ...
                     sin(angle(H(ii)) - angle(Ht(ii) - angle(wIH(ii,jj))));
          end
        end
        
        % a little bit different for the second part of w
        for ii = 1:sizeH
          for jj = 1:sizeI           
            deltaEwHI2(ii,jj) =  (1- abs(H(ii)).^2).' .* ( (abs(H(ii)) - abs(Ht(ii))) .* ...
                     cos(angle(H(ii)) - angle(Ht(ii))) ).' * abs(zI(jj)) .* ...
                     sin(angle(H(ii)) - angle(Ht(ii)) - angle(wIH(ii,jj))).' + ...
                     abs(H(ii)) * abs(Ht(ii)) * sin(angle(H(ii)) - angle(Ht(ii))) .* ...
                     (abs(zI(jj)) / atanh(abs(H(ii)))).* ...
                     cos(angle(H(ii)) - angle(Ht(ii) - angle(wIH(ii,jj))));
          end
        end

        wHI1 = abs(wIH) - k * deltaEwHI1;
        wOH1 = abs(wHO) - k * deltaEwOH1;
        
        wHI2 = angle(wIH) - k * deltaEwHI2;
        wOH2 = angle(wHO) - k * deltaEwOH2;
        
        wIH = wHI1 .* exp(1).^(1i.* wHI2);
        wHO = wOH1 .* exp(1).^(1i.* wOH2);
        
        % storing zO into O
        O(r, :) = zO;  
    end
        er_matrix(counter) = er;
        counter = counter +1;
        disp(er) 
end

% Printing the Error-Value \ Iteration graph
% y = (1:counter-1);
% figure
% plot(y, er_matrix)
% title('ER Value Development')
% xlabel('Iteration')
% ylabel('ER Value')
% axis([0 counter 0 inf]);

% size(T)
% size(O)
% figure; plot(1:length(T),abs(T),'g', 1:length(O), abs(O),'r');
figure; plotconfusion(abs(T),abs(O),"Complex Valued Output");

end