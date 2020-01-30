x1 = [0+0i 0+0i 0+1i 0+1i 0+1i 1+0i 1+1i 1+1i 0+0i 0+0i 0+1i 1+0i 1+0i 1+0i 1+1i 1+1i 1+0i];
x2 = [0+0i 0+1i 0+0i 0+1i 1+0i 1+0i 0+1i 1+1i 1+0i 1+1i 1+1i 0+0i 0+1i 1+1i 0+0i 1+0i 1+0i];

X = [x1; x2];

% T = [1+0i 0+1i 0+0i 1+1i 0+1i 1+1i 0+1i 1+0i 0+1i 0+0i 0+0i 0+0i 0+1i 0+0i 0+0i 0+1i ];
T =  [1 1i 0 1+1i 1i 1+1i 1i 1 1i 0 0 0 1i 0 0 1i ];

% Run neural network
[wIH, wHO, O] = neural_network_xor(X, T);


