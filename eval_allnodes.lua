require 'torch'
require 'nn'

opt = {
  dataset = 'simple',
  nClasses = 88,
  nThreads = 8,
  batchSize = 16,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  cudnn = 1,
  model = 'experiment1/checkpoints/hierachical_training/iter16000_net.t7',
  ntest = math.huge,
  randomize = 0,
  cropping = 'center',
  data_root = '/do_not_store/sunil/abhi',
  data_list = '/do_not_store/sunil/abhi/test_labels.json',
  mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224}
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- load in network
assert(opt.model ~= '', 'no model specified')
print('loading ' .. opt.model)
local net = torch.load(opt.model)
net:evaluate()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

-- ship to GPU
if opt.gpu > 0 then
  input = input:cuda()
  net:cuda()
end

-- eval
local accuracies = torch.DoubleTensor(opt.nClasses, 2):fill(0)
local cmp_tensor = torch.DoubleTensor(opt.batchSize, opt.nClasses):fill(0.5)
local overall_correct = 0
local overall_misspredict = 0
local counter = 0
local maxiter = math.floor(math.min(data:size(), opt.ntest) / opt.batchSize)

local match = 0
local missmatch = 0
local totallabels = data:size() * opt.nClasses
local perclassmatch = torch.LongTensor(1, opt.nClasses):fill(0)
local totalperclasslabels = data:size()

for iter = 1, maxiter do
  collectgarbage()
  
  local data_im,labels = data:getBatch()

  input:copy(data_im)
  local pred_outputs = net:forward(input)
  local outputs = pred_outputs:double()
  outputs = torch.gt(outputs, cmp_tensor)

  -- for perfect match
  eq_vector = torch.eq(outputs:byte(), labels:byte())

  match = match + eq_vector:sum()
  perclassmatch = torch.add(perclassmatch,  eq_vector:sum(1):long())


  -- for perfect match end here

  -- for hit and miss calculation
  --[[
  correct_labels = labels and outputs
  incorrect_labels = torch.ne(labels:byte(), outputs:byte())
  
  overall_correct = overall_correct + correct_labels:sum()
  overall_misspredict = overall_misspredict +  incorrect_labels:sum()

  for i=1, opt.batchSize do
    for j=1, opt.nClasses do
      if(outputs[i][j] == 1 or labels[i][j] == 1) then
        if outputs[i][j] == labels[i][j] then
          accuracies[j][1] = accuracies[j][1] +1
        else
          accuracies[j][2] = accuracies[j][2] +1
        end
      end
    end
  end
  -- for hit and miss calculation end here
  --]]

  --[[
  for i=1, opt.batchSize do
    print(outputs[i]:nonzero())
    print(labels[i]:nonzero())
  end
  --]]

  --[[
  local _,preds = output:float():sort(2, true)

  for i=1,opt.batchSize do
    local rank = torch.eq(preds[i], data_label[i]):nonzero()[1][1]
    if rank == 1 then
      top1 = top1 + 1
    end
    if rank <= 5 then
      top5 = top5 + 1
    end
  end

  --]]

  counter = counter + opt.batchSize
  
  --[[
  print(('%s: Eval [%8d / %8d]:\t Hit: %6d, Miss: %6d, Hit(Per): %.4f, Miss(Per): %.4f'):format(
    opt.model, iter, maxiter,
    overall_correct, overall_misspredict, 
    overall_correct/(overall_correct+overall_misspredict), 
    overall_misspredict/(overall_correct+overall_misspredict)))
  --]]

  print(('Overall Hit: %6d, Miss: %6d: Hit(Per): %.4f, Miss(Per): %.4f'):format(
    match, (iter*opt.batchSize*opt.nClasses) - match, 
    match/((iter*opt.batchSize*opt.nClasses)), 
    ((iter*opt.batchSize*opt.nClasses)-match)/((iter*opt.batchSize*opt.nClasses))))
end

--[[
print(('Overall Hit: %6d, Miss: %6d: Hit(Per): %.4f, Miss(Per): %.4f'):format(
  overall_correct, overall_misspredict, 
  overall_correct/(overall_correct+overall_misspredict), 
  overall_misspredict/(overall_correct+overall_misspredict)))

for i=1, opt.nClasses do
  print(('Class: %4d, Hit: %6d, Miss: %6d, Hit(Per): %.4f, Miss(Per): %.4f'):format(
    i, accuracies[i][1], accuracies[i][2],
    accuracies[i][1]/(accuracies[i][1]+accuracies[i][2]),
    accuracies[i][2]/(accuracies[i][1]+accuracies[i][2])))
end
--]]

print(('Overall Hit: %6d, Miss: %6d: Hit(Per): %.4f, Miss(Per): %.4f'):format(
  match, (totallabels - match), 
  match/(totallabels), 
  (totallabels-match)/(totallabels)))

for i=1, opt.nClasses do
  print(('Class: %4d, Hit: %6d, Miss: %6d, Hit(Per): %.4f, Miss(Per): %.4f'):format(
    i, perclassmatch[1][i], (totalperclasslabels-perclassmatch[1][i]),
    perclassmatch[1][i]/(totalperclasslabels),
    ((totalperclasslabels-perclassmatch[1][i])/(totalperclasslabels))))
end
