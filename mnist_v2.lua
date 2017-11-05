require 'dp'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'

-- Load the mnist data set
ds = dp.Mnist()

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw'):cuda()
trainTargets = ds:get('train', 'targets', 'b'):cuda()
validInputs = ds:get('valid', 'inputs', 'bchw'):cuda()
validTargets = ds:get('valid', 'targets', 'b'):cuda()
testInputs = ds:get('test', 'inputs', 'bchw'):cuda()
testTargets = ds:get('test', 'targets', 'b'):cuda()

-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 

module:cuda()
-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion()
criterion:cuda()


optimState = {
    learningRate = 1e-1,
  }
  
-- a confusion matrix
cm = optim.ConfusionMatrix(10)
-- create a function to compute 
function classEval(module, inputs, targets)
   cm:zero()
   for idx=1,inputs:size(1) do
      local input, target = inputs[idx], targets[idx]
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end
batch_size = 10
epoch_size = 100
--print(trainInputs:size())
--x = trainInputs[{{1,100}}]
--print(x:size())
require 'dpnn'
function trainEpoch(module, criterion, inputs, targets, batch_size, epoch_size,
    valid_inputs, valid_targets)
    --declare number of epochs outside function
    bestAccuracy, bestEpoch = 0, 0
    wait = 0
    
    for k = 1,epoch_size do 
      print(k)
     --random shuffle of inputs and targets
     local shuffle = torch.randperm(inputs:size(1))
     --local idx = math.random(1,inputs:size(1))
     for i = 1, shuffle:size(1) do
        input = inputs:cuda()
        target = targets:cuda()
        input[i], target[i] = inputs[shuffle[i]], targets[shuffle[i]]
      end
      -- subset the inputs and targets by batch size and update based on batch
      for i=0,inputs:size(1)-batch_size,batch_size do
        local batch_input = input[{{i+1, i+batch_size}}]
        local batch_target = target[{{i+1, i+batch_size}}]
        parameters, gradParameters = module:getParameters()
        
        local function feval(parameters)
          gradParameters:zero()
          --local batch_input = input[{{i+1, i+batch_size}}]
          --local batch_target = target[{{i+1, i+batch_size}}]
          -- forward
          local output = module:forward(batch_input)
          local loss = criterion:forward(output, batch_target)
          -- backward
          local gradOutput = criterion:backward(output, batch_target)
          local gradInput = module:backward(batch_input, gradOutput)
          return loss, gradParameters
        end    
        optim.adagrad(feval, parameters, optimState)
      end
    
     local validAccuracy = classEval(module, valid_inputs, valid_targets)
     if validAccuracy > bestAccuracy then
        bestAccuracy, bestEpoch = validAccuracy, k
        --torch.save("/path/to/saved/model.t7", module)
        print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
        wait = 0
     else
        wait = wait + 1
        if wait > 5 then break end
     end
    end
end

 
--for epoch=1,30 do
trainEpoch(module, criterion, trainInputs, trainTargets,batch_size,epoch_size,
  validInputs, validTargets)

testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))
