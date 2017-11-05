require 'dp'
require 'nn'
require 'cutorch'
require 'cunn'
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
module:add(nn.LogSigmoid())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 

module:cuda()
-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion()
criterion:cuda()


require 'optim'
-- allocate a confusion matrix
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
epoch_size = 30
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
        -- forward
        local output = module:forward(batch_input)
        local loss = criterion:forward(output, batch_target)
        -- backward
        local gradOutput = criterion:backward(output, batch_target)
        module:zeroGradParameters()
        local gradInput = module:backward(batch_input, gradOutput)
        -- update
        module:updateGradParameters(0.9) -- momentum (dpnn)
        module:updateParameters(0.1) -- W = W - 0.1*dL/dW
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

 

gpuTime1 = sys.clock()
--for epoch=1,30 do
trainEpoch(module, criterion, trainInputs, trainTargets,batch_size,epoch_size,
  validInputs, validTargets)

gpuTime2 = sys.clock()

print(gpuTime2 - gpuTime1)

testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))


output = module:forward(testInputs)
outputClass = torch.Tensor(output:size(1),1):cuda()

  for i = 1,output:size(1) do
    val, ind = torch.max(output[i],1)
    outputClass[i] = ind[1]
    if outputClass[i] == 10 then
      outputClass[i] = 0
    end
  end
  
  
for k = 1,10 do
  tab = {}
  for i = 1,outputClass:size(1) do
    if outputClass[i][1] == k then
      table.insert(tab, i)
    end
  end
  
  outputSelect = torch.Tensor(table.getn(tab), output:size(2)):cuda()
  targetSelect = torch.Tensor(table.getn(tab)):cuda()
  for i = 1, outputSelect:size(1) do
    outputSelect[i] = output[tab[i]]
    targetSelect[i] = testTargets[tab[i]]
  end
  
  conf = torch.Tensor(table.getn(tab)):cuda()
  for i = 1,outputSelect:size(1) do
    a, b = torch.sort(outputSelect[i],true)
    conf[i] = a[1]-a[2]
  end
  
  val, ind = torch.sort(conf, 1, true)
  --outputSort = torch.Tensor(table.getn(tab),1)
  targetSort = torch.Tensor(table.getn(tab)):cuda()
  
  for i = 1, outputSelect:size(1) do
    targetSort[i] = targetSelect[ind[i]]
  end
  
  predTrue = torch.Tensor(targetSort:size(1)):cuda()
  goodPred = {}
  badPred = {}
  for i = 1, targetSort:size(1) do
    if targetSort[i] == k then
      predTrue[i]=1
      table.insert(goodPred, 1)
    else
      predTrue[i]=0
      table.insert(badPred, 1)
    end
  end
  goodPred1 = torch.Tensor(goodPred):cuda()
  badPred1 = torch.Tensor(badPred):cuda()
  good = 1/torch.sum(goodPred1)
  bad = 1/torch.sum(badPred1)
  
  x = torch.Tensor(predTrue:size(1)+1):cuda() 
  y = torch.Tensor(predTrue:size(1)+1):cuda()
  x[1], y[1] = 0, 0
  for i = 1, predTrue:size(1) do
    if predTrue[i] ==1 then
      x[i+1] = good+x[i]
      y[i+1] = y[i]
    else
      x[i+1] = x[i]
      y[i+1] = bad+y[i]
    end
  end
  
  require 'gnuplot'
  gnuplot.figure(k)
  gnuplot.plot(y,x)
  gnuplot.axis{0,1,0,1}
  --gnuplot.title(print("ROC for class ", k))
  print(k)
  local area = 0
  for i = 2, x:size(1) do
    area = area + (y[i] - y[i-1]) * (x[i-1] + x[i])/2
  end
  print(area)
end


  
