clear all;  clc; close all;

img = double(imread('./img.png'));
img = img / 255;
y = img(:);

Num_iter = 1000;
lam = 0.1;
cost = zeros(1, Num_iter);
N = length(y);
I = speye(N);
D = I(2:N, :) - I(1:N-1, :);
DDT = D * D';

x = y;
Dx = D*x;
Dy = D*y;

for k = 1:Num_iter
    F = sparse(1:N-1, 1:N-1, abs(Dx)/lam) + DDT;
    x = y - D' * (F\Dy);
    Dx = D*x;
    cost(k) = 0.5 * sum(abs(x-y).^2) + lam*sum(abs(Dx));
end

x = reshape(x, [300, 332]);
y = reshape(y, [300, 332]);
figure()
imshow(y);
figure()
imshow(x);