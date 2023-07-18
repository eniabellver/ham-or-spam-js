const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const fs = require('fs');

const CSV_FILE_PATH = './files/sms_spam_dataset.csv';

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

function padSequences(sequences, maxLength) {
  return sequences.map((sequence) => {
    const padding = Array(maxLength - sequence.length).fill(0);
    const sequenceWithPadding = sequence.concat(padding);
    return sequenceWithPadding.map((val) => Number(val));
  });
}

function createSequences(tokenizedTextData, wordIndex) {
  return tokenizedTextData.map((tokens) =>
    tokens.map((token) => wordIndex[token] || 0)
  );
}

function createOneHotEncodedSequences(tokenizedTextData, wordIndex, maxLength) {
  return tokenizedTextData.map((tokens) => {
    const sequence = tokens.map((token) => wordIndex[token] || 0);
    const indices = tf.tensor1d(sequence, 'int32');
    const oneHotEncodedSequence = tf.oneHot(indices, Object.keys(wordIndex).length);
    const paddedSequence = tf.pad(
      oneHotEncodedSequence,
      [[0, maxLength - sequence.length], [0, 0]],
      1
    );
    return paddedSequence;
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
    const vocabularyArray = Array.from(vocabulary).filter(Boolean);
    const wordIndex = createWordIndex(vocabulary);
    const sequences = createSequences(tokenizedTextData, wordIndex);
    const maxLength = Math.max(...sequences.map((sequence) => sequence.length));
    const oneHotEncodedSequences = createOneHotEncodedSequences(
      tokenizedTextData,
      wordIndex,
      maxLength
    );

    // tensors
    const textTensor = tf.stack(oneHotEncodedSequences);
    const labelTensor = tf.tensor2d(numericLabelData, [
      numericLabelData.length,
      1,
    ]);
    const normalizedTextTensor = textTensor.div(tf.scalar(vocabularyArray.length - 1));

    // compile
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [maxLength, vocabularyArray.length] }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
      loss: 'binaryCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy'],
    });

    // train
    await model.fit(normalizedTextTensor, labelTensor, { epochs: 10 })

    // example predictions
    console.log('Example Predictions:');
    const predictions = model.predict(normalizedTextTensor);
    const predictionsData = predictions.array();``
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
    return { vocabularyArray, wordIndex, maxLength, model };

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
