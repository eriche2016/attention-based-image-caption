require 'nn'
require 'nngraph'

require 'misc.LinearTensorD3'
-- require 'misc.probe' -- for debugger on nngraph module, put the layer to check gradient and outputs 
-- require 'misc.utils_bg' -- also for debugger purpose 

local LSTM_Armour = {} 

-- class CAddTableD2D3
local CAddTableD2D3, parent = torch.class('nn.CAddTableD2D3', 'nn.Module')
function CAddTableD2D3:__init() 
    parent.__init(self) 
    self.gradInput = {} 
end 

function CAddTableD2D3:updateOutput(input) 
    -- input[1]: bz * Dh 
    -- input[2]: bz * L * Dh
    assert(type(input) == 'table' and #input == 2) 
    local hProj, xProj = unpack(input)
    assert(hProj:dim()==2 and xProj:dim()==3) 

    self.output:resizeAs(xProj)
    self.output:copy(xProj)
    
    local L = xProj:size(2)  -- xProj1, xProj2, ..., where index i means the i-th time step  

    for k = 1, L do 
        self.output:narrow(2, k, 1):add(hProj) 
    end 

    return self.output 
end 

function CAddTableD2D3:updateGradInput(input, gradOutput) 
    -- gradOutput: [bz * L * Dh] 
    assert(gradOutput:dim() == 3)
    local hProj, xProj = unpack(input) 

    for i = 1, #input do 
        self.gradInput[i] = self.gradInput[i] or input[i].new() 
        self.gradInput[i]:resizeAs(input[i])
    end 
    
    -- same reference 
    grad_h, grad_x = unpack(self.gradInput) 
    grad_h:fill(0)

    local L = xProj:size(2) 
   
    -- accumulated gradients w.r.t the hiddent state at previsous timestep: h_{i-1}
    for t = 1, L do 
        grad_h:add(gradOutput:narrow(2, t, 1))    
    end 
    
    -- just copy the gradients w.r.t input x_{1, 2, 3, ..., L} for attention based model at each time step 
    grad_x:copy(gradOutput)

    return self.gradInput 
end 

-- Custome Alpha View: [bz * L] --> [bz * L * 1]
local CustomAlphaView, parent = torch.class('nn.CustomAlphaView', 'nn.Module') 
function CustomAlphaView:__init() 
    parent.__init(self)
end 

function CustomAlphaView:updateOutput(input) 
    assert(input:dim() == 2) 
    self.output = input:view(input:size(1), input:size(2), 1)
    return self.output 
end 

function CustomAlphaView:updateGradInput(input, gradOutput)
   
    self.gradInput = gradOutput:view(gradOutput:size(1), gradOutput:size(2))
    
    return self.gradInput 
end 


function LSTM_Armour.makeWeightedSumUnit() 
    -- note each sample in the batch may has different alignments(or called weights)
    local alpha = nn.Identity()()              -- bz * L 
    local alphaMatrix = nn.CustomAlphaView()(alpha) -- bz * L * 1 
    
    local x = nn.Identity()()                      -- bz * L * xDim
    local g = nn.MM(true, false)({x, alphaMatrix}) -- bz * xDim * 1 

    g = nn.Select(3, 1)(g)                          -- bz * xDim 
    local inputs, outputs = {x, alpha}, {g}

    -- return a nn.Module 
    return nn.gModule(inputs, outputs)
end 


-- input_size1: the embeddings of the word vector 
-- input_size2: the embeddings of the image
function LSTM_Armour.lstm(input_size1, input_size2, output_size, rnn_size, n, dropout) 
  dropout = dropout or 0 

  -- there will be 2*n+2 inputs
  
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- word embeddings
  table.insert(inputs, nn.Identity()()) -- extra input for the image features extracted from the image

  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x1, x2, input_size_L1, input_size_L2
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+2]
    local prev_c = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then 
      x1 = inputs[1]  -- word vector 
      x2 = inputs[2]  -- glimpse   
      input_size_L1 = input_size1 
      input_size_L2 = input_size2
    else  -- currently only 1 layer, this is not modified 
       x1 = outputs[(L-1)*2]  -- lower layer output: next_h
       x2 = inputs[2]  -- will always be the image features vectors, feed to the new layer of LSTM 

      if dropout > 0 then x1 = nn.Dropout(dropout)(x1):annotate{name='drop_' .. L} end -- apply dropout, if any
      if dropout > 0 then x2 = nn.Dropout(dropout)(x2):annotate{name='drop_' .. L} end -- apply dropout, if any(may use different dropout rate in the futures)

      input_size_L1 = rnn_size
      input_size_L2 = input_size2 
    end

    -- evaluate the input sums at once for efficiency
    local w2h = nn.Linear(input_size_L1, 4 * rnn_size)(x1):annotate{name='w2h_'..L} -- word embedding input 
    -- to avoid double bias terms
    local v2h = nn.Linear(input_size_L2, 4 * rnn_size, false)(x2):annotate{name='v2h_'..L} -- visual input 
	
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate{name='h2h_'..L}

    local all_input_sums = nn.CAddTable()({w2h, v2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)

    -- 2 instead of 1 because it supports batch input
    -- split method is a node method which will return 4 new nodes
    -- because nn.SplitTable(2)() will return 4 output nodes 
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   
    -- next is 'current', which will be used as input at the next timestep
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  
  -- add the glimpse and the input vectors
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

-- Attention unit: consider x_1, x_2, x_3, ..., x_L is the feature sequence of the image 
function LSTM_Armour.Attention_Unit(hDim, xDim, aDim) 
	local prev_h = nn.Identity()()                     -- bz * hDim  
	local x = nn.Identity()()                          --  bz * L * D, ie bz * 196 * 512
	local hProj = nn.Linear(hDim, aDim)(prev_h)        -- bz * aDim 
	local xProj = nn.LinearTensorD3(xDim, aDim)(x)     -- bz * L * aDim  
	local sumTanh = nn.Tanh()(nn.CAddTableD2D3()({hProj, xProj})) -- bz * L * aDim 
	local vProj = nn.LinearTensorD3(aDim, 1)(sumTanh)  -- bz * L * 1 
	local alpha = nn.SoftMax()(nn.Select(3, 1)(vProj)) -- bz * L  

	local inputs, outputs = {prev_h, x}, {alpha}
	return nn.gModule(inputs, outputs)
end 

-- make LSTM with attention mechannism of selecting from the image features set: x = {x_1, x_2, ..., x_L}
-- note that here  we just set input_size2 has to be  512, we will come to this and set it as a parameters 
-- in the future 
function LSTM_Armour.LSTM_with_Attention(input_size1, input_size2, output_size, rnn_size, attSize, depth, dropout) 
    dropout = dropout or 0
	depth = depth or 1

	assert(depth==1, 'currently only support 1 layer of LSTM') 
	
	local prev_c = nn.Identity()()	
	local prev_h = nn.Identity()() 

	local word_vector = nn.Identity()() 
	local x = nn.Identity()()                           --- bz * L * D, ie, bz * 196 * 512
    -- alignment 
	local alpha = LSTM_Armour.Attention_Unit(rnn_size, input_size2, attSize)({prev_h, x})   

    -- soft attention, glimpse
	local g = LSTM_Armour.makeWeightedSumUnit()({x, alpha})  
     
    -- rnn is the output
    -- currently input_size2 must be 512
    local rnn = LSTM_Armour.lstm(input_size1, input_size2, output_size, rnn_size, depth, dropout)({word_vector, g, prev_c, prev_h})
	

	local inputs, outputs = {word_vector, x, prev_c, prev_h}, {rnn}

	return nn.gModule(inputs, outputs)
end 

return LSTM_Armour
