console.log('Hello World');
import {MnistData} from './data.js';

async function showWorkspace(){
  // Create a container in the visor
    tfvis.visor();  
}
async function showExamples(data) {
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas 
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer s
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    model.add(tf.layers.flatten());
  
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Training', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}


// load Data program
var data = new MnistData();
async function loadData() {  
    await data.load()
    await showExamples(data);
  }
//show Model
const trainModel = getModel();
async function showModel() {  
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, trainModel);
}
//train Data
async function trainData() {  
    await train(trainModel, data);
    // await showAccuracy(trainModel, data)
  }
async function showAccuracyFunc(){
  await showAccuracy(trainModel, data)
}
async function showConfMatrix(){
  await showConfusion(trainModel, data);
}


//Train Button
var visorButton = document.getElementById('show-visor-button');
var visorButtonid = "show-visor-button"
visorButton.setAttribute("id", visorButtonid)
$("#show-visor-button").click(function(e){
  showWorkspace()
  document.getElementById(visorButtonid).disabled = 'true';
});
//Load Data
var dataButton = document.getElementById('load-data-button');
var dataButtonid = "load-data-button"
dataButton.setAttribute("id", dataButtonid)
$("#load-data-button").click(function(e){
  loadData()
  document.getElementById(dataButtonid).disabled = 'true';
});
//Model Architecture 
var modelButton = document.getElementById('model-button');
var modelButtonid = "model-button"
modelButton.setAttribute("id", modelButtonid)
$("#model-button").click(function(e){
  showModel()
  document.getElementById(modelButtonid).disabled = 'true';
  var hid = document.getElementsByClassName("model1");
    // Emulates jQuery $(element).is(':hidden');
    if(hid[0].offsetWidth > 0 && hid[0].offsetHeight > 0) {
        hid[0].style.visibility = "visible";
        hid[0].style.opacity = "100%";
    }
});
//Train Data
var trainButton = document.getElementById('train-data-button');
var trainButtonid = "train-data-button"
trainButton.setAttribute("id", trainButtonid)
$("#train-data-button").click(function(e){
  trainData()
  document.getElementById(trainButtonid).disabled = 'true';
});
//Show Accuracy
var accuracyButton = document.getElementById('show-accuracy-button');
var accuracyButtonid = "show-accuracy-button"
accuracyButton.setAttribute("id", accuracyButtonid)
$("#show-accuracy-button").click(function(e){
  showAccuracyFunc()
  document.getElementById(accuracyButtonid).disabled = 'true';
});
//Show Confusion Matrix
var confMatrix = document.getElementById('confusion-matrix-button');
var confMatrixid = "confusion-matrix-button"
confMatrix.setAttribute("id", confMatrixid)
$("#confusion-matrix-button").click(function(e){
  showConfMatrix()
  document.getElementById(confMatrixid).disabled = 'true';
});




