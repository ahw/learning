const webpack = require('webpack')

module.exports = {
    entry: './src/main.js',
    output: {
        path: './dist',
        filename: 'main.js',
        publicPath: 'https://localhost:8082/'
    },
    module: {
        loaders: [{
            test: /\.js$/,
            exclude: /node_modules/,
            loader: 'babel-loader'
        }]
    },
    devtool: 'source-map'
    // Instead of hard-coding this Uglify plugin configuration, you can just
    // run webpack with the -p or --optimize-minimize flag which will minify
    // stuff.
    // plugins: [
    //     // Turn on for production minification
    //     // new webpack.optimize.UglifyJsPlugin({
    //     //     compress: {
    //     //         warnings: false,
    //     //     },
    //     //     output: {
    //     //         comments: false,
    //     //     }
    //     // })
    // ]
 }