var mongoose = require("mongoose");
var passportLocalMongoose = require("passport-local-mongoose");

var UserSchema = new mongoose.Schema({
    

    name: String,
    username: String,
    passsword: String,
    number: Number,
    email: String,
    level: Number,
    exuno: String,
    exu: String,
    microno: String,
    micro: String,
    bloodno:String,
    blood: String,
    result: [
      {
         type: mongoose.Schema.Types.ObjectId,
         ref: "Result"
      }
      
      ],
   
    });

UserSchema.plugin(passportLocalMongoose);


module.exports = mongoose.model("User", UserSchema);