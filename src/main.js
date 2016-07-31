import Network from './network'
import Matrix from './matrix'

console.log('This is main.js')
let n = new Network([10, 10, 10])
// let n = new Network([784, 100, 10])
// n.feedforward([[2], [3]])


function createData(length) {
    return Array.from({ length }).map(() => {
        let m = Matrix.rand(10, 1, { lo: 0, hi: 100 })

        let max = m.max()
        let median = m.median()
        let n = Matrix.create(10, 1, (i, j) => {
            if (i === median.row && j === median.col) {
                return 1
            } else {
                return 0
            }
        })

        return [m, n]
    })
}

let trainingData = createData(5000)
let testData = createData(20)

n.sgd({
    epochs: 10,
    miniBatchSize: 20,
    eta: 5.0,
    trainingData,
    testData
})
