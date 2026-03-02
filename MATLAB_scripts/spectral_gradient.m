function spectral_gradient(input_file, m, n, output_prefix)
% SPECTRAL_GRADIENT Compute x and y gradients using spectral differentiation
%
%   spectral_gradient(input_file, m, n, output_prefix)
%
%   Inputs:
%       input_file    - Path to input binary file (float32, row-major)
%       m             - Number of rows
%       n             - Number of columns
%       output_prefix - Prefix for output files (will append _fft_grad_x.raw, _fft_grad_y.raw)
%
%   Example:
%       spectral_gradient('field.raw', 512, 512, 'field')

    % Read binary input (float32, row-major order)
    fid = fopen(input_file, 'rb');
    if fid == -1
        error('Cannot open input file: %s', input_file);
    end
    data = fread(fid, m * n, 'float32');
    fclose(fid);

    % Reshape to m x n (row-major: n is the fast dimension)
    field = reshape(data, [n, m])';

    % Compute wavenumbers (grid spacing = 1)
    kx = (2 * pi / n) * [0:n/2-1, -n/2:-1];
    ky = (2 * pi / m) * [0:m/2-1, -m/2:-1];

    [KX, KY] = meshgrid(kx, ky);

    % FFT of the field
    field_hat = fft2(field);

    % Spectral differentiation: multiply by i*k
    grad_x_hat = 1i * KX .* field_hat;
    grad_y_hat = 1i * KY .* field_hat;

    % Inverse FFT to get gradients
    grad_x = real(ifft2(grad_x_hat));
    grad_y = real(ifft2(grad_y_hat));

    % Write output files (float32, row-major)
    output_grad_x = sprintf('%s_fft_grad_x.raw', output_prefix);
    output_grad_y = sprintf('%s_fft_grad_y.raw', output_prefix);

    fid = fopen(output_grad_x, 'wb');
    fwrite(fid, grad_x', 'float32');
    fclose(fid);

    fid = fopen(output_grad_y, 'wb');
    fwrite(fid, grad_y', 'float32');
    fclose(fid);

    fprintf('Wrote gradient files:\n  %s\n  %s\n', output_grad_x, output_grad_y);
end
