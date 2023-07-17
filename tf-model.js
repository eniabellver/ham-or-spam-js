const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const fs = require('fs');
const natural = require('natural');

const csvFilePath = './files/sms_spam_dataset.csv';

const textData = [];
const labelData = [];

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

    const vocabulary = new Set();
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
          const prediction =
            predictions.arraySync()[i][0] > 0.5 ? 'spam' : 'ham';
          console.log(`Text: ${text}`);
          console.log(`Label: ${label}`);
          console.log(`Prediction: ${prediction}`);
          console.log('-----------------------------');
        }
      })
      .catch((error) => {
        console.error('Error occurred during model training:', error);
      });
  });

function getLabelCategory(label) {
  return label === '1' ? 'spam' : 'ham';
}
