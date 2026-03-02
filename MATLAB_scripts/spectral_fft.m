function spectral_fft(input_file, m, n, output_prefix)
% SPECTRAL_FFT Compute 2D FFT and store compactly using conjugate symmetry
%
%   spectral_fft(input_file, m, n, output_prefix)
%
%   Inputs:
%       input_file    - Path to input binary file (float32, row-major)
%       m             - Number of rows
%       n             - Number of columns
%       output_prefix - Prefix for output file (will append _fft.raw)
%
%   Output format (exploits Hermitian symmetry of real input FFT):
%       - Diagonal (row == col): real part
%       - Bottom-left (row > col): real part
%       - Top-right (row < col): imaginary part
%
%   Example:
%       spectral_fft('field.raw', 512, 512, 'field')

    % Read binary input (float32, row-major order)
    fid = fopen(input_file, 'rb');
    if fid == -1
        error('Cannot open input file: %s', input_file);
    end
    data = fread(fid, m * n, 'float32');
    fclose(fid);

    % Reshape to m x n (row-major: n is the fast dimension)
    field = reshape(data, [n, m])';

    % Compute FFT
    field_hat = fft2(field);

    % Create compact output using conjugate symmetry
    output = zeros(m, n, 'single');

    for i = 1:m
        for j = 1:n
            if i == j
                % Diagonal: real part
                output(i, j) = real(field_hat(i, j));
            elseif i > j
                % Bottom-left: real part
                output(i, j) = real(field_hat(i, j));
            else
                % Top-right: imaginary part
                output(i, j) = imag(field_hat(i, j));
            end
        end
    end

    % Write output file (float32, row-major)
    output_file = sprintf('%s_fft.raw', output_prefix);

    fid = fopen(output_file, 'wb');
    fwrite(fid, output', 'float32');
    fclose(fid);

    fprintf('Wrote FFT file:\n  %s\n', output_file);
end
