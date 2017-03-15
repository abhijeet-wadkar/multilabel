require 'torch'
require 'nn'

opt = {
  epoch = 5,
  dataset = 'simple',
  nClasses = 88,
  nInternalNodes = 20,
  nThreads = 16,
  batchSize = 16,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  cudnn = 1,
  models = 'all_nodes/checkpoints',
  ntest = math.huge,
  randomize = 0,
  cropping = 'center',
  data_root = '/remote_hdd3/Koni/abhi',
  data_list = '/remote_hdd3/Koni/abhi/hierarchical_all_node_labels/test_labels.json',
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

for counter=1, opt.epoch do

  local model = opt.models .. 'iter' .. counter .. '_net.t7'

  -- create data loader
  local DataLoader = paths.dofile('data/data.lua')
  local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
  print("Dataset: " .. opt.dataset, " Size: ", data:size())

  -- load in network
  assert(opt.model ~= '', 'no model specified')
  print('loading ' .. model)
  local net = torch.load(model)
  net:evaluate()

  -- create the data placeholders
  local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

  -- ship to GPU
  if opt.gpu > 0 then
    input = input:cuda()
    net:cuda()
  end

  -- eval
  local top1 = 0
  local top5 = 0
  local counter = 0
  local maxiter = math.floor(math.min(data:size(), opt.ntest) / opt.batchSize)

  for iter = 1, maxiter do
    collectgarbage()
    
    local data_im,data_labels = data:getBatch()
    input:copy(data_im)

    local outputs = net:forward(input)

    outputs = outputs:narrow(2, opt.nInternalNodes+1, opt.nClasses-opt.nInternalNodes)
    sorted_outputs, preds = outputs:sort(2, true)
    data_labels = data_labels:narrow(2, opt.nInternalNodes+1, opt.nClasses-opt.nInternalNodes)
    data_labels = data_labels:nonzero()

    for i=1,opt.batchSize do
      local rank = torch.eq(preds[i]:byte(), data_labels[i][2]):nonzero()[1][1]
      if rank == 1 then
        top1 = top1 + 1
      end
      if rank <= 5 then
        top5 = top5 + 1
      end
    end

    counter = counter + opt.batchSize

    print(('Testing %s: Eval [%8d / %8d]:\t Top1: %.4f  Top5: %.4f'):format(
        opt.model, iter, maxiter,
        top1/counter, top5/counter))
    
  end
end