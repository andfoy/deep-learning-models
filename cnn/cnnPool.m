function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
% poolDim - dimension of pooling region
% convolvedFeatures - convolved features to pool (as given by cnnConvolve)
% convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
% pooledFeatures - matrix of pooled features in the form
% pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
resultDim = floor(convolvedDim / poolDim);

pooledFeatures = zeros(resultDim, resultDim, numFilters, numImages);

% Instructions:
% Now pool the convolved features in regions of poolDim x poolDim,
% to obtain the
% (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages
% matrix pooledFeatures, such that
% pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the
% value of the featureNum feature for the imageNum image pooled over the
% corresponding (poolRow, poolCol) pooling region.
%
% Use mean pooling here.

%%% YOUR CODE HERE %%%
for featureNum = 1:numFilters
    for imageNum = 1:numImages
        for poolRow = 1:resultDim
            offsetRow = 1 + (poolRow-1) * poolDim;
            for poolCol = 1:resultDim
                offsetCol = 1 + (poolCol-1) * poolDim;
                patch = convolvedFeatures(offsetRow:offsetRow + poolDim-1, ...
                        offsetCol:offsetCol+poolDim-1,featureNum, imageNum);
                pooledFeatures(poolRow, poolCol, featureNum, imageNum) = mean(patch(:));
            end
        end
    end
end    


end
 
