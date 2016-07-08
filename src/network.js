'use strict'

let jStat = require('jstat').jStat
let sprintf = require('sprintf').sprintf
let math = require('mathjs')

function sigmoid(z) {
    if (typeof z === 'number') {
        return 1/(1+Math.pow(Math.E, -z))
    } else {
        return z.map(sigmoid)
    }
}

function printMatrix(m, label) {
    label = label || ""
    let pad = label.split("").map(s => ' ').join("")
    return m.map((row, i) => {
        if (!Array.isArray(row)) {
            row = [row]
        }
        return (i ? pad : label) + '[ ' + row.map(e => sprintf('%0.03f', e)).join(', ') + ' ]'
    }).join('\n')
}


class Network {

    constructor(sizes) {
        this.sizes = sizes

        // biases is an array of bias vectors. The input layer obviously
        // does not have a bias vector which is why we are slicing sizes
        // from index 1. biases[i] = [ [b1], [b2], ..., [bN] ], which is the
        // bias vector for layer i. The bias vector is a column vector which
        // is why it is an array of single-element arrays.
        //
        // jStat.rand(rows, cols) creates a matrix of the given row and
        // column dimensions where each element is pulled from the Normal
        // distribution.
        this.biases = this.sizes.slice(1).map(layerSize => jStat.rand(layerSize, 1))

        // weights is an array of weight matrices. There is one weight
        // matrix "between" each layer. For a given weight matrix
        // weights[i], the column indices map to the different neurons in
        // the incoming "X" layer; the row indices map to the different
        // neurons in the output "Y" layer. Example:
        //
        // weights[i] = [
        //  [ w_x_0y_0, w_x_1y_0, w_x2_y0 ],
        //  [ w_x_0y_1, w_x_1y_1, w_x2_y1 ],
        //  [ w_x_0y_2, w_x_1y_2, w_x2_y2 ]
        // ]
        //
        // w_x2_y1 should be interpreted as, "the weight of the connection
        // between neuron x2 in the X input layer and neuron y1 in the Y
        // output layer."
        //
        // See comment above regarding jStat.rand()
        this.weights = this.sizes.slice(0, -1).map((xLayerSize, i) => {
            let yLayerSize = this.sizes[i + 1]
            return jStat.rand(yLayerSize, xLayerSize)
        })
    }

    feedforward(a) {
        for (let i = 0; i < this.sizes.length - 1; i++) {
            console.log(`Iteration ${i+1}`)
            console.log(`${printMatrix(this.weights[i], 'w = ')}\n${printMatrix(a, 'x = ')}\n${printMatrix(this.biases[i], 'b = ')}`)
            let b = this.biases[i]
            let w = this.weights[i]
            let wa = math.multiply(w, a).map(m => [m])
            console.log(printMatrix(wa, 'wa = '))
            a = math.add(wa, b)
            a = a.map(row => row[0])
            console.log(printMatrix(a, 'a = '))
            // a = sigmoid(jStat(w).multiply(a).add(b))
            // console.log(jStat(w).multiply(a))
            // console.log(printMatrix(a, 'a = '))
        }

        return a
    }
}

let n = new Network([2, 3, 4, 2])
n.feedforward([2, 3])
console.log('biases')
console.log(n.biases)
console.log('weights')
console.log(n.weights)
console.log(sigmoid([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))