classdef DWT
    
    % wavelet transform without the agonizing pain
    %
    % basically it takes care of the junk behind the
    % scenes so that dwt/idwt is as easy as fft/ifft
    %
    % lax about input shapes (vectorized ok) for pcg
    %
    % Example
    %   x = rand(1,8);
    %   W = DWT(size(x),'db2');
    %   y = W * x; % forward
    %   z = W'* y; % inverse
    %   norm(x-z)
    %     ans = 4.2819e-13
   
    properties (SetAccess = private)
        sizeINI
        filters
        mode
        dec
        trans = 0 % transpose flag (0 or 1)
    end

    methods
        
        %% constructor
        function obj = DWT(sizeINI,wname,mode)
            
            if nargin<1
                error('sizeINI is required');
            end
            if nargin<2
                wname = 'sym3'; % orthogonal
            end
            if nargin<3
                mode = 'per'; % periodic
            end

            if ~isnumeric(sizeINI) || ~isvector(sizeINI)
                error('sizeINI must be the output of size()');
            end
            while numel(sizeINI) && sizeINI(end)==1
                sizeINI(end) = []; % remove trailing ones
            end
            if any(sizeINI==0) || numel(sizeINI)>3
                error('only 1d, 2d or 3d supported');
            end
            if numel(sizeINI)<=2 && mod(max(sizeINI),2)
                error('only even dimensions supported');
            elseif numel(sizeINI)>2 && any(mod(sizeINI,2))
                error('only even dimensions supported');
            end
            obj.sizeINI = reshape(sizeINI,1,[]);
            
            [LoD HiD LoR HiR] = wfilters(wname);
            obj.filters.LoD = LoD;
            obj.filters.HiD = HiD;
            obj.filters.LoR = LoR;
            obj.filters.HiR = HiR;
            
            obj.mode = mode;
            
        end
        
        %% y = W*x or y = W'*x
        function y = mtimes(obj,x)
            
            % allow looping over coil dimension
            nc = numel(x) / prod(obj.sizeINI);
            
            if mod(nc,1)
                error('size(x) not compatible with sizeINI [%s]',num2str(obj.sizeINI));
            end

            if nc>1
                
                x = reshape(x,prod(obj.sizeINI),nc);
                y = zeros(size(x),'like',x);
                
                for c = 1:nc
                    y(:,c) = reshape(obj * x(:,c),[],1);
                end
                
                y = reshape(y,[obj.sizeINI nc]);
                
            else
                
                % need correct shape
                x = reshape(x,[obj.sizeINI 1]);
                
                if obj.trans==0
                    
                    % forward transform
                    if isvector(x)
                        [CA CD] = dwt(x,obj.filters.LoD,obj.filters.HiD,'mode',obj.mode);
                        if isrow(x); y = [CA CD]; else; y = [CA;CD]; end
                    elseif ndims(x)==2
                        [CA CH CV CD] = dwt2(x,obj.filters.LoD,obj.filters.HiD,'mode',obj.mode);
                        y = [CA CV; CH CD];
                    elseif ndims(x)==3
                        wt = dwt3(x,{obj.filters.LoD,obj.filters.HiD,obj.filters.LoR,obj.filters.HiR},'mode',obj.mode);
                        y = cell2mat(wt.dec);
                    else
                        error('only 1d, 2d or 3d supported');
                    end
                    
                else
                    
                    % inverse transform
                    if isvector(x)
                        if isrow(x)
                            C = mat2cell(x,1,[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        else
                            C = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],1);
                        end
                        y = idwt(C{1},C{2},obj.filters.LoR,obj.filters.HiR,'mode',obj.mode);
                    elseif ndims(x)==2
                        C = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(1)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2]);
                        y = idwt2(C{1},C{2},C{3},C{4},obj.filters.LoR,obj.filters.HiR,'mode',obj.mode);
                    elseif ndims(x)==3
                        wt.sizeINI = obj.sizeINI;
                        wt.filters.LoD = {obj.filters.LoD,obj.filters.LoD,obj.filters.LoD};
                        wt.filters.HiD = {obj.filters.HiD,obj.filters.HiD,obj.filters.HiD};
                        wt.filters.LoR = {obj.filters.LoR,obj.filters.LoR,obj.filters.LoR};
                        wt.filters.HiR = {obj.filters.HiR,obj.filters.HiR,obj.filters.HiR};
                        wt.mode = obj.mode;
                        wt.dec = mat2cell(x,[obj.sizeINI(1)/2 obj.sizeINI(2)/2],[obj.sizeINI(2)/2 obj.sizeINI(2)/2],[obj.sizeINI(3)/2 obj.sizeINI(3)/2]);
                        y = idwt3(wt);
                    else
                        error('only 1d, 2d or 3d supported');
                    end
                    
                end
                
            end
            
        end
        
        %% detect W' and set flag
        function obj = ctranspose(obj)
            
            obj.trans = ~obj.trans;
            
        end
        
    end
    
end