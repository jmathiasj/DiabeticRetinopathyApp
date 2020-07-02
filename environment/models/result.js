var mongoose = require("mongoose");
mongoose.Promise = global.Promise;

var resultSchema = mongoose.Schema({
    level: Number,
    exudatess: String,
    bloodvess: String,
    microaneus: String,
    exunes: String,
    microses:String,
    bldnes: String,
    datet:String,
    author: {
        id: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User"
        },
        username: String
    }
});

module.exports = mongoose.model("Result", resultSchema);