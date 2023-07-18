const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const fs = require('fs');

const CSV_FILE_PATH = './files/sms_spam_dataset.csv';

//we set a max to avoid OOM issues
const MAX_VOCABULARY_SIZE = 5000;
const MAX_SEQUENCE_LENGTH = 100;

// helper functions
function tokenizeText(text) {
  return text.toLowerCase().split(/\s+/);
}

function createWordIndex(vocabulary) {
  const wordIndex = {};
  vocabulary.forEach((word, index) => {
    wordIndex[word] = index;
  });
  return wordIndex;
}

function createSequences(tokenizedTextData, wordIndex) {
  return tokenizedTextData.map((tokens) =>
    tokens.map((token) => wordIndex[token] || 0)
  );
}

function padSequence(sequence, maxLength) {
  if (sequence.length >= maxLength) {
    return sequence.slice(0, maxLength);
  } else {
    const padding = Array(maxLength - sequence.length).fill(0);
    return sequence.concat(padding);
  }
}


function createOneHotEncodedSequences(tokenizedTextData, wordIndex) {
  return tokenizedTextData.map((tokens) => {
    const sequence = tokens
      .map((token) => wordIndex[token] || wordIndex['<UNK>'])
      .slice(0, MAX_SEQUENCE_LENGTH);
    const paddedSequence = padSequence(sequence, MAX_SEQUENCE_LENGTH);
    const indices = tf.tensor1d(paddedSequence, 'int32');
    const oneHotEncodedSequence = tf.oneHot(indices, MAX_VOCABULARY_SIZE);
    return oneHotEncodedSequence;
  });
}

function getLabelCategory(prediction, threshold = 0.5) {
  if (prediction > threshold) {
    return 'spam';
  } else {
    return 'ham';
  }
}
/*-------------------------------------------------------*/

async function processCsvFile() {
  return new Promise((resolve, reject) => {
    const textData = [];
    const labelData = [];
    const vocabulary = new Set();

    fs.createReadStream(CSV_FILE_PATH)
      .pipe(csv())
      .on('data', (row) => {
        const tokens = tokenizeText(row.sms);
        textData.push(row.sms);
        labelData.push(parseInt(row.label));
        tokens.forEach((token) => vocabulary.add(token));
      })
      .on('end', () => {
        console.log('CSV file successfully processed');
        resolve({ textData, labelData, vocabulary });
      })
      .on('error', reject);
  });
}

async function trainModel() {
  try {
    const { textData, labelData, vocabulary } = await processCsvFile();

    // prep data
    const numericLabelData = labelData.map((label) => parseInt(label));
    const tokenizedTextData = textData.map(tokenizeText);
    const vocabularyArray = Array.from(vocabulary)
      .filter(Boolean)
      .slice(0, MAX_VOCABULARY_SIZE);
    vocabularyArray.push('<UNK>');
    const wordIndex = Object.fromEntries(
      vocabularyArray.map((word, index) => [word, index])
    );
    const sequences = createSequences(tokenizedTextData, wordIndex);
    const paddedSequences = sequences.map((sequence) =>
      padSequence(sequence, MAX_SEQUENCE_LENGTH)
    );


    // tensors
    const textTensor = tf.tensor2d(paddedSequences);
    const labelTensor = tf.tensor2d(numericLabelData, [
      numericLabelData.length,
      1,
    ]);
    const normalizedTextTensor = textTensor.div(tf.scalar(MAX_VOCABULARY_SIZE - 1));

    // compile
    const model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: MAX_VOCABULARY_SIZE,
        outputDim: 16,
        inputLength: MAX_SEQUENCE_LENGTH,
      })
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
      loss: 'binaryCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy'],
    });

    // train
    await model.fit(normalizedTextTensor, labelTensor, { epochs: 10 });

    // example predictions
    console.log('Examples:');
    const predictions = model.predict(textTensor);
    const predictionsData = predictions.arraySync();
    for (let i = 0; i < 5; i++) {
      const text = textData[i];
      const label = labelData[i];
      const prediction = predictionsData[i][0];
      const category = getLabelCategory(label, prediction);
      console.log(`Text: ${text}`);
      console.log(`Label: ${label}`);
      console.log(`Prediction: ${category}`);
      console.log('-----------------------------');
    }

    // export
    return { vocabularyArray, wordIndex, model };

  } catch (error) {
    console.error('FAILURE - Error occurred during model training:', error);
    return null;
  }
}

if (require.main === module) {
  trainModel();
}

module.exports = trainModel;

//////////* old working code before refactor *//////////
/*
 fs.createReadStream(csvFilePath)
   .pipe(csv())
   .on('data', (row) => {
     textData.push(row.sms);
     labelData.push(parseInt(row.label));
   })
   .on('end', () => {
     console.log('CSV file successfully processed');
     //prep data
     const numericLabelData = labelData.map((label) => parseInt(label));
     const tokenizedTextData = textData.map((text) =>
       text.toLowerCase().split(/\s+/)
     );
     tokenizedTextData.forEach((tokens) =>
       tokens.forEach((token) => vocabulary.add(token))
     );
     const vocabularyArray = Array.from(vocabulary);
     const wordIndex = {};
     vocabularyArray.forEach((word, index) => {
       wordIndex[word] = index;
     });
     const sequences = tokenizedTextData.map((tokens) =>
       tokens.map((token) => wordIndex[token] || 0)
     );
     const maxLength = Math.max(...sequences.map((sequence) => sequence.length));
     const paddedSequences = sequences.map((sequence) => {
       const padding = Array(maxLength - sequence.length).fill(0);
       return sequence.concat(padding);
     });
     // create tensors
     const textTensor = tf.tensor2d(paddedSequences);
     const labelTensor = tf.tensor2d(numericLabelData, [
       numericLabelData.length,
       1,
     ]);
     const normalizedTextTensor = textTensor.div(
       tf.scalar(vocabularyArray.length - 1)
     );
     // compile
     const model = tf.sequential();
     model.add(
       tf.layers.embedding({
         inputDim: vocabularyArray.length,
         outputDim: 16,
         inputLength: maxLength,
       })
     );
     model.add(tf.layers.flatten());
     model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
     model.compile({
       loss: 'binaryCrossentropy',
       optimizer: 'adam',
       metrics: ['accuracy'],
     });
     // train
     model
       .fit(normalizedTextTensor, labelTensor, { epochs: 10 })
       .then((history) => {
         console.log('Model training complete');
         // predict
         const predictions = model.predict(normalizedTextTensor);
         console.log('Example Predictions:');
         for (let i = 0; i < 5; i++) {
           const text = textData[i];
           const label = labelData[i];
           const prediction = predictions.arraySync()[i][0];
           const category = getLabelCategory(label, prediction);
           console.log(`Text: ${text}`);
           console.log(`Label: ${label}`);
           console.log(`Prediction: ${category}`);
           console.log('-----------------------------');
         }
       })
       .catch((error) => {
         console.error('Error occurred during model training:', error);
       });
});
*/
