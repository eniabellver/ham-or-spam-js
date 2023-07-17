const tf = require('@tensorflow/tfjs-node');
const csv = require('csv-parser');
const fs = require('fs');

const csvFilePath = './files/SMSSpamCollection.csv';

const textData = [];
const labelData = [];

fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    textData.push(row.text);
    labelData.push(row.label);
  })
  .on('end', () => {
    console.log('CSV file successfully processed');

    // convert data to tensors
    const textTensor = tf.tensor2d(textData, [textData.length, 1]);
    const labelTensor = tf.tensor2d(labelData, [labelData.length, 1]);

    // normalize text tensor
    const normalizedTextTensor = textTensor.div(tf.scalar(255));

    // create and compile the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // train the model
    model
      .fit(normalizedTextTensor, labelTensor, {
        epochs: 10,
        batchSize: 32,
        callbacks: tf.node.tensorBoard('logs'),
      })
      .then((history) => {
        const prediction = model.predict(normalizedTextTensor);

        console.log('Prediction:', prediction);
      });
  });
