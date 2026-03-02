function central_diff_gradient(input_file, m, n, output_prefix, order, boundary)
% CENTRAL_DIFF_GRADIENT Compute x and y gradients using central difference
%
%   central_diff_gradient(input_file, m, n, output_prefix, order, boundary)
%
%   Inputs:
%       input_file    - Path to input binary file (float32, row-major)
%       m             - Number of rows
%       n             - Number of columns
%       output_prefix - Prefix for output files (will append _grad_x.raw, _grad_y.raw)
%       order         - Accuracy order: 2, 4, 6, or 8 (default: 2)
%       boundary      - Boundary condition (default: 'circular'):
%                       'circular'  - Periodic boundaries
%                       'symmetric' - Mirror/reflect at boundaries
%                       'replicate' - Extend edge values
%                       'zero'      - Zero padding
%
%   Kernel sizes by order:
%       order 2: 3-point stencil
%       order 4: 5-point stencil
%       order 6: 7-point stencil
%       order 8: 9-point stencil
%
%   Example:
%       central_diff_gradient('field.raw', 512, 512, 'field', 4, 'circular')

    if nargin < 5
        order = 2;
    end
    if nargin < 6
        boundary = 'circular';
    end

    % Central difference coefficients for first derivative
    switch order
        case 2
            coeff = [-1/2, 0, 1/2];
        case 4
            coeff = [1/12, -2/3, 0, 2/3, -1/12];
        case 6
            coeff = [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60];
        case 8
            coeff = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280];
        otherwise
            error('Unsupported order: %d. Use 2, 4, 6, or 8.', order);
    end

    % Read binary input (float32, row-major order)
    fid = fopen(input_file, 'rb');
    if fid == -1
        error('Cannot open input file: %s', input_file);
    end
    data = fread(fid, m * n, 'float32');
    fclose(fid);

    % Reshape to m x n (row-major: n is the fast dimension)
    field = reshape(data, [n, m])';

    % Compute gradients using manual convolution
    half_width = (length(coeff) - 1) / 2;
    grad_x = zeros(m, n);
    grad_y = zeros(m, n);

    for k = 1:length(coeff)
        offset = k - half_width - 1;

        % X gradient (shift in column direction)
        % Negate offset for correct convolution direction: df/dx = (f[j+1] - f[j-1]) / 2
        shifted_x = shift_array(field, 0, -offset, boundary);
        grad_x = grad_x + coeff(k) * shifted_x;

        % Y gradient (shift in row direction)
        shifted_y = shift_array(field, -offset, 0, boundary);
        grad_y = grad_y + coeff(k) * shifted_y;
    end

    % Write output files (float32, row-major)
    output_grad_x = sprintf('%s_cd_grad_x.raw', output_prefix);
    output_grad_y = sprintf('%s_cd_grad_y.raw', output_prefix);

    fid = fopen(output_grad_x, 'wb');
    fwrite(fid, grad_x', 'float32');
    fclose(fid);

    fid = fopen(output_grad_y, 'wb');
    fwrite(fid, grad_y', 'float32');
    fclose(fid);

    fprintf('Wrote gradient files (order %d, %s boundary):\n  %s\n  %s\n', ...
            order, boundary, output_grad_x, output_grad_y);
end

function out = shift_array(arr, row_shift, col_shift, boundary)
% SHIFT_ARRAY Shift array with specified boundary condition
    [m, n] = size(arr);

    switch boundary
        case 'circular'
            row_idx = mod((0:m-1) - row_shift, m) + 1;
            col_idx = mod((0:n-1) - col_shift, n) + 1;
            out = arr(row_idx, col_idx);

        case 'symmetric'
            row_idx = (1:m) - row_shift;
            col_idx = (1:n) - col_shift;
            row_idx = reflect_index(row_idx, m);
            col_idx = reflect_index(col_idx, n);
            out = arr(row_idx, col_idx);

        case 'replicate'
            row_idx = (1:m) - row_shift;
            col_idx = (1:n) - col_shift;
            row_idx = max(1, min(m, row_idx));
            col_idx = max(1, min(n, col_idx));
            out = arr(row_idx, col_idx);

        case 'zero'
            out = zeros(m, n);
            src_row = max(1, 1+row_shift):min(m, m+row_shift);
            src_col = max(1, 1+col_shift):min(n, n+col_shift);
            dst_row = src_row - row_shift;
            dst_col = src_col - col_shift;
            out(dst_row, dst_col) = arr(src_row, src_col);

        otherwise
            error('Unsupported boundary: %s', boundary);
    end
end

function idx = reflect_index(idx, len)
% REFLECT_INDEX Reflect indices at boundaries (symmetric)
    idx = idx - 1;  % 0-based
    idx = mod(idx, 2*len);
    idx(idx >= len) = 2*len - 1 - idx(idx >= len);
    idx = idx + 1;  % back to 1-based
end
