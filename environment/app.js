//**********************************************************Node Modules********************************************************************************************************
var compression = require('compression');
var http = require('http');
var base64 = require('base-64');
var datetime = new Date();
const image2base64 = require('image-to-base64');
var express = require("express");
var fs = require("fs");
var app = express();
var bodyParser = require("body-parser");
var nodemailer = require("nodemailer");
var passport = require("passport");
var flash = require('express-flash-messages');
var passportLocalMongoose = require("passport-local-mongoose");
var LocalStrategy = require("passport-local").Strategy;
var User = require("./models/user");
var mongoose = require("mongoose");
var doctor = require("./models/doctor");
var admin = require("./models/admin");
// var al_quiz = require("./models/alcoholQue");
// var co_quiz = require("./models/cocaineQue");
// var he_quiz = require("./models/heroinQue");
// var ec_quiz = require("./models/ecstasyQue");
// var ma_quiz = require("./models/marijuanaQue");
var rehab = require("./models/rehab");
var Comment =require("./models/comments");
var Result= require("./models/result")
var methodOverride =require('method-override');

var all_score = {
    level: Number,
    exuno: String,
    exu: String,
    microno: String,
    micro: String,
    bloodno:String,
    blood: String,};
    
  all_score = {
    level: 0,
    exuno: "",
    exu: "",
    microno: "",
    micro: "",
    bloodno:"",
    blood: "",
            };
            
app.use(function(req, res, next) {
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next();
});
app.use(compression());

app.use(express.static("./static")); 
  
app.use(function(req,res,next){
    res.locals.currentUser = req.user;
    next();
}) ; 

//**************Send contact*************************************************************//
// app.post('/home', function (req, res) {
//   let transporter = nodemailer.createTransport({
//       host: 'smtp.gmail.com',
//       port: 465,
//       secure: true,
//       auth: {
//           user: 'j.mathjen@gmail.com',
//           pass: 'Sun@riSe1998'
//       }
//   });

//   let mailOptions = {
//       from:"j.mathjen@gmail.com",
//       to: "j.mathjen@gmail.com", // receiver Email
//       subject: req.body.Subject, // Subject line
//       body: req.body.Message, // plain text body
//       html: req.body.Email +'<b>has a doubt</b>' // html body
//   };
//       transporter.sendMail(mailOptions, (error, info) => {
//       if (error) {
//           return console.log(error);
//       }
//       console.log("Message send Successfully");
    
//   });
//  });





//*******************************************************************Node Mailer*******************************************************************************************************************************************************************************

// var transporter = nodemailer.createTransport({ // Creating a Transporter(The admin account from which ails will be sent )
//     service: 'gmail',
//     auth: {
//         user: 'mathias.jenellm@gmail.com',
//         pass: 'Sun@riSe1998'
//     }
// });
var transporter = nodemailer.createTransport({
      host: 'smtp.gmail.com',
      port: 465,
      secure: true,
      auth: {
          user: '',
          pass: ''
      }
  });



//************************************************************Extra Variable Methods*******************************************************************************************************

mongoose.connect("mongodb://localhost:27017/cupcake"); // set the database named cupcake
 // tell app to use body parse
app.set("view engine", "ejs");
app.use(express.static('public'));
app.use(bodyParser.json({limit: '10mb'}));
app.use(bodyParser.urlencoded({limit:'10mb', extended: true }));
app.use(methodOverride('_method'));
app.use(flash());
//************************************************************ Admin Database Schema*******************************************************************************************************
// admin.create({
//                   username: "admin",
//                   password: "admin",
          
         
            
//                     },function(err,admin){
//                         if(err){
//                             console.log(err);
//                         }else{
//                             console.log("Admin");
//                             console.log(admin);
//                         }
//                     });

//  co_quiz.create({
//                   question:"Do u find difficulty in getting the thought of drinking out f your mind?",
//                   opt1:"Never",
//                   opt2:"Sometimes", 
//                   opt3:"Often", 
//                   opt4:"Nearly Always ", 
//                   opt5:"Always"      
            
//                     },function(err,ccquiz){
//                         if(err){
//                             console.log(err);
//                         }else{
//                             console.log("New Quiz Question created:");
//                             console.log(ccquiz);
//                         }
//                     });
        

//  he_quiz.create({
//                   question:"Do u find difficulty in getting the thought of drinking out f your mind?",
//                   opt1:"Never",
//                   opt2:"Sometimes", 
//                   opt3:"Often", 
//                   opt4:"Nearly Always ", 
//                   opt5:"Always"      
            
//                     },function(err,ccquiz){
//                         if(err){
//                             console.log(err);
//                         }else{
//                             console.log("New Quiz Question created:");
//                             console.log(ccquiz);
//                         }
//                     });

//  ec_quiz.create({
//                   question:"Do u find difficulty in getting the thought of drinking out f your mind?",
//                   opt1:"Never",
//                   opt2:"Sometimes", 
//                   opt3:"Often", 
//                   opt4:"Nearly Always ", 
//                   opt5:"Always"      
            
//                     },function(err,ccquiz){
//                         if(err){
//                             console.log(err);
//                         }else{
//                             console.log("New Quiz Question created:");
//                             console.log(ccquiz);
//                         }
//                     });
                
 
 
//   ma_quiz.create({
//                   question:"Do u find difficulty in getting the thought of drinking out f your mind?",
//                   opt1:"Never",
//                   opt2:"Sometimes", 
//                   opt3:"Often", 
//                   opt4:"Nearly Always ", 
//                   opt5:"Always"      
            
//                     },function(err,ccquiz){
//                         if(err){
//                             console.log(err);
//                         }else{
//                             console.log("New Quiz Question created:");
//                             console.log(ccquiz);
//                         }
//                     });
        
 
//********************************************************************************************************************************************************************************************************** 
//PASSPORT CONFIGURATION
app.use(require("express-session")({
    secret: "Drug",
    resave: false,
    saveUninitialized: false
}));
app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStrategy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());


passport.serializeUser(function(User, done) {
  // placeholder for custom user serialization
  // null is for errors
  done(null, User);
});

passport.deserializeUser(function(User, done) {
  // placeholder for custom user deserialization.
  // maybe you are going to get the user from mongo by id?
  // null is for errors
  done(null, User);
});

app.use(function(req,res,next){
   res.locals.currentUser  = req.user;
   next();
});

// app.use(function(req,res,next){
//   res.locals.currentadmin  = req.admin;
//   next();
// });



//************************************************************cONNECTIONS*******************************************************************************************************
// "/" => home.ejs..........it is the homepage
app.get("/",function(req,res){
    
    
    res.render("home");
})

app.get("/admin",function(req,res){
    
    res.render("admin");
})

app.get("/home/:id/trial",isLoggedIn,function(req,res){
    
    
    res.render("trial");
})



// "/addict" => addict.ejs..........it is the page which has info abt various addiction
app.get("/addict",function(req,res){
    
    
    res.render("addict");
})
app.get("/userprofile",function(req,res){
    
    
  res.render("userprofile",{user:User});
})
app.get("/symptomsandcauses",function(req,res){
    
    
    res.render("symptomsandcauses");
})

// "/treatment" => treatment.ejs..........it is the page which has info abt various treament facilities
app.get("/treatment",isLoggedIn, function(req,res){
    if(req.user){
    
    
    res.render("treatment");
    }
    else{
        res.redirect('/login');
    }
});


// "/enroll" => enroll.ejs..........it is the page which will hv the enrollment to online counciling
app.get("/enroll",function(req,res){
    
    
    res.render("enroll");
})

// "/aboutus" => aboutUs.ejs..........it is the page which will take to info pg about us
app.get("/aboutus",function(req,res){
    
    
    res.render("aboutUs");
})

app.get("/forgot",function(req,res){
    
    
    res.render("forgot");
});

app.post("/forgot",function (req, res){
    
    console.log("u hit forgot post");
    var email = req.body.email;
    
    User.find({'email':email},function(err,us){
                
                        if(err){
                            console.log(err);
                        }
                        else{
                                    us.forEach(function(user){
                                        
                                        
                                        
                                                  var mailOptions = {
                                                     from: 'j.mathjen@gmail.com',
                                                     to: us.email,
                                                     subject:'Forgot Pw',
                                                     html: '<p>hello</p>'+user.username+'<p>,</p>'+'<h3>We received your request</h3>'+'<p><b>Your Password:</b></p>'+user.password
                                                };
                                 
                                 
                                                  transporter.sendMail(mailOptions, function(error, info) {
                                                           
                                                            if (error) {
                                                                console.log(error);
                                                                req.flash('notify','Sending email Failed');
                                                            }
                                                            else {
                                                                console.log('Email sent: ' + info.response);
                                                               req.flash('notify','Email sent successfully');
                                                                res.redirect("/sendemail");
                                                            }
                                                    });
                                  
                                        
                                    });
                                              
                  
                            
                        }
                
        
        
        
        
        
    });

    
   
});

app.get("/userprofile/:id",function(req,res){
   
    User.findById(req.params.id).populate("result").exec(function(err,user){
        if(err){
            console.log(err);
        } else {
         var datetime = new Date();
        console.log(datetime);
            console.log(user);
             res.render("disp",{user:user});
        }
        
        
                                  
                                        
                                    });
    
    

});
app.get("/report/:id",function(req,res){
   
    Result.findById(req.params.id).exec(function(err,result){
        if(err){
            console.log(err);
        } else {
  
            console.log(result);
             res.render("reports",{result:result});
        }
        
        
                                  
                                        
                                    });
    
    

});






app.post('/home/:id/trial', function (req, res) { 

var opt = req.body.pred;

postData = JSON.stringify({
    'image': opt
});

var options = {
    hostname: '0.0.0.0',
    port: 5000,
    path: '/prediction',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
};

var reqs = http.request(options, (res) => {
    var data = '';
    res.on('data', (chunk) => {
        data += chunk.toString(); // buffer to string
    });
    res.on('end', () => {
        
        data = JSON.parse(data);
        
        all_score.level=data.prediction;
        req.user.level=data.prediction;
        console.log(data.prediction);
        console.log(data.exudates);
        console.log(data.bloodves);
        console.log(data.microaneu);
        console.log(data.exune);
        console.log(data.microse);
        console.log(data.bldne);

        User.findById(req.params.id,function(err, user){
       if(err){
           console.log(err);
           res.redirect('/home/'+user._id);
       } else {
           var dates=new Date();
           datel=dates.toString();
        Result.create({level:data.prediction,
            exudatess:data.exudates,
            bloodvess:data.bloodves,
            microaneus:data.microaneu,
            exunes:data.exune,
            microses:data.microse,
            bldnes:data.bldne,
            datet:datel
        }, function(err, result){
           if(err){
               console.log(err);
           } else {
               //add username and id to comment
               result.author.id = req.user._id;
               result.author.username = req.user.username;
               //save comment
               result.save();
               user.result.push(result);
               user.save();
               console.log(user);
               console.log(result);
              
              
              
           }
        });
       }
   });
        

        console.log('No more data in response.');
    });
});

reqs.on('error', (e) => {
    console.error(`problem with request: ${e.message}`);
});
console.log(req.user.level);
reqs.write(postData);


res.redirect("/home/" +req.user._id);


reqs.end();  

 });


 



app.get("/reset",function(req,res){
    
    
    res.render("reset");
})

app.post('/reset', function (req, res) {
    if (!req.session.reset) return res.end('reset token not set');
    
    var password = req.body.password;
    var confirm = req.body.confirm;
    if (password !== confirm) return res.end('passwords do not match');
    
    
    
    //forgot.expire(req.session.reset.id);
    delete req.session.reset;
    res.end('password reset');
});

//==============================================================Doctor view and Adding=======================================================================

app.get("/viewdoctor",function(req,res){
    
    doctor.find(function(err, alldoctor){
        
        if(err){
            console.log(err);
        }else{
            
          res.render("viewDoctor",{doctor:alldoctor});  
        }
    });
    
});

app.get("/adddoctor",function(req,res){
    
   res.render("adddoctor");
})

app.post("/adddoctor",function(req,res){
    
    var name = req.body.name;
    var Qualification = req.body.Qualification;
    var Specialization = req.body.Specialization;
    var Skypeid = req.body.Skypeid;
    var image = req.body.image;
    doctor.create({
                  name:name,
                  Qualification:Qualification,
                  Specilaization:Specialization,
                  Skypeid:Skypeid,
                  image:image
         
            
                    },function(err,doctor){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New Doctor created:");
                            console.log(doctor);
                            res.redirect("/adddoctor");
                        }
                        
                        
                    });
   
    
})


//=============================================================Send Promotional Emails========================================================================== 


app.get("/sendemail",function(req,res){
    
  res.render("sendEmail");
})

app.post("/sendemail",function(req,res){


    
    var Subject = req.body.Subject;
    var Body = req.body.Body;
    
     User.find(function(err,user){
         
         if(err){
             
             console.log(err);
         }
         else{
              
              user.forEach(function(us){
                      
                            var mailOptions = {
                                     from: 'j.mathjen@gmail.com',
                                     to: us.email,
                                     subject: Subject,
                                     text: Body
                     };
                 
                 
                     transporter.sendMail(mailOptions, function(error, info) {
                           
                            if (error) {
                                console.log(error);
                                req.flash('notify','Sending email Failed');
                            }
                            else {
                                console.log('Email sent: ' + info.response);
                               req.flash('notify','Email sent successfully');
                                res.redirect("/sendemail");
                            }
                     });
                  
                  
                  
                  
              });
             
             
             
         }
    
         
         

        
});
    
     

   
})









//==============================================================Rehab view and Adding=======================================================================

app.get("/viewRehab",function(req,res){
    var noMatch;
    if(req.query.search){
        
       const regex = new RegExp(escapeRegex(req.query.search), 'gi');
        rehab.find({Location: regex},function(err, allRehab){
                
                if(err){
                    console.log(err);
                }else{
                    
                    if(allRehab.length<1){
                        noMatch = "No Rehabs found in that location, Sorry.";
                    }
                  res.render("viewRehab",{rehab:allRehab, noMatch:noMatch});  
                }
            });
        
    }else{
             rehab.find(function(err, allRehab){
                
                if(err){
                    console.log(err);
                }else{
                    
                  res.render("viewRehab",{rehab:allRehab, noMatch: noMatch});  
                }
            });
        }
          
});

app.get("/addRehab",function(req,res){
    
   res.render("addRehab");
})

app.post("/addRehab",function(req,res){
    
    var Name = req.body.Name;
    var Location = req.body.Location;
    var Contact = req.body.Contact;
    var image = req.body.image;
    rehab.create({
                  Name:Name,
                  Location:Location,
                  Contact:Contact,
                  image:image
         
            
                    },function(err,rehab){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New Rehab created:");
                            console.log(rehab);
                            res.redirect("/addRehab");
                        }
                    });
   
    
})



//======================================================Adding quiz questions===================================================================

app.get("/add_alQue",function(req,res){
    
   res.render("add_alQue");
})

app.post("/add_alQue",function(req,res){
    
    var question = req.body.question;
    var opt1 = req.body.opt1;
    var opt2 = req.body.opt2;
    var opt3 = req.body.opt3;
    var opt4 = req.body.opt4;
    var opt5 = req.body.opt5;
    al_quiz.create({
                    question: question ,
                    opt1: opt1,
                    opt2: opt2,
                    opt3: opt3,
                    opt4: opt4,
                    opt5: opt5
                     
            
                    },function(err,info){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New alcohol quiz question created:");
                            console.log(info);
                            res.redirect("/add_alQue");
                        }
                    });
   
    
})



app.get("/add_coQue",function(req,res){
    
   res.render("add_coQue");
})

app.post("/add_coQue",function(req,res){
    
    var question = req.body.question;
    var opt1 = req.body.opt1;
    var opt2 = req.body.opt2;
    var opt3 = req.body.opt3;
    var opt4 = req.body.opt4;
    var opt5 = req.body.opt5;
    co_quiz.create({
                    question: question ,
                    opt1: opt1,
                    opt2: opt2,
                    opt3: opt3,
                    opt4: opt4,
                    opt5: opt5
                     
            
                    },function(err,info){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New cocaine quiz question created:");
                            console.log(info);
                            res.redirect("/add_coQue");
                        }
                    });
   
    
})

app.get("/add_ecQue",function(req,res){
    
   res.render("add_ecQue");
})

app.post("/add_ecQue",function(req,res){
    
    var question = req.body.question;
    var opt1 = req.body.opt1;
    var opt2 = req.body.opt2;
    var opt3 = req.body.opt3;
    var opt4 = req.body.opt4;
    var opt5 = req.body.opt5;
    ec_quiz.create({
                    question: question ,
                    opt1: opt1,
                    opt2: opt2,
                    opt3: opt3,
                    opt4: opt4,
                    opt5: opt5
                     
            
                    },function(err,info){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New ecstasy quiz question created:");
                            console.log(info);
                            res.redirect("/add_ecQue");
                        }
                    });
   
    
})

app.get("/add_maQue",function(req,res){
    
   res.render("add_maQue");
})

app.post("/add_maQue",function(req,res){
    
    var question = req.body.question;
    var opt1 = req.body.opt1;
    var opt2 = req.body.opt2;
    var opt3 = req.body.opt3;
    var opt4 = req.body.opt4;
    var opt5 = req.body.opt5;
    ma_quiz.create({
                    question: question ,
                    opt1: opt1,
                    opt2: opt2,
                    opt3: opt3,
                    opt4: opt4,
                    opt5: opt5
                     
            
                    },function(err,info){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New marijuana quiz question created:");
                            console.log(info);
                            res.redirect("/add_maQue");
                        }
                    });
   
    
})

app.get("/add_heQue",function(req,res){
    
   res.render("add_heQue");
})

app.post("/add_heQue",function(req,res){
    
    var question = req.body.question;
    var opt1 = req.body.opt1;
    var opt2 = req.body.opt2;
    var opt3 = req.body.opt3;
    var opt4 = req.body.opt4;
    var opt5 = req.body.opt5;
    he_quiz.create({
                    question: question ,
                    opt1: opt1,
                    opt2: opt2,
                    opt3: opt3,
                    opt4: opt4,
                    opt5: opt5
                     
            
                    },function(err,info){
                        if(err){
                            console.log(err);
                        }else{
                            console.log("New heroin quiz question created:");
                            console.log(info);
                            res.redirect("/add_heQue");
                        }
                    });
   
    
})


//=====================================================================================================================================
//quiz
//=====================================================================================================================================
app.get("/alcoholq",function(req,res){
  
     al_quiz.find({},function(err,al_que){
         
         if(err){
             console.log(err);
         }else
         {      
                 res.render("alcohol",{que:al_que});
      
         }
     
         
      });
     
 });
 
app.post("/alcoholq",function(req,res){
     var name= req.body.score;
     
     all_score.quiz1=name;
     console.log(all_score);
     
     
     console.log("U hit alcohol quiz");
     console.log("name:");
     console.log(name);
     
     res.redirect("/quizhome");
     
     
 });

app.get("/cocaineq",function(req,res){
  
     co_quiz.find({},function(err,al_que){
         
         if(err){
             console.log(err);
         }else
         {      
                 res.render("cocaine",{que:al_que});
      
         }
     
         
      });
     
 });
 
app.post("/cocaineq",function(req,res){
     var name= req.body.score;
     
     all_score.quiz4=name;
     console.log(all_score);
     
     
     console.log("U hit cocaine quiz");
     console.log("name:");
     console.log(name);
     
     res.redirect("/quizhome");
     
     
 });
 


//=========================
//Comment Routes
//=========================
app.get("/viewdoctor/:id",function(req,res){
   
    doctor.findById(req.params.id).populate("comments").exec(function(err,doctor){
        if(err){
            console.log(err);
        } else {
            
            console.log(doctor);
             res.render("display", {doctor:doctor});
        }
    
    
});
});




app.get("/viewdoctor/:id/comments/new",isLoggedIn,  function(req, res){
    // find doctor by id
    
console.log(req.params.id);
    doctor.findById(req.params.id, function(err, alldoctor){
        if(err){
            console.log(err);
        } else {
             res.render("new-comments", {doctor: alldoctor});
        }
    });
});


app.post("/viewdoctor/:id/comments",isLoggedIn,function(req, res){
   //lookup doctor using ID
   doctor.findById(req.params.id,function(err, doctor){
       if(err){
           console.log(err);
           res.redirect("/viewdoctor");
       } else {
        Comment.create(req.body.comment, function(err, comment){
           if(err){
               console.log(err);
           } else {
               //add username and id to comment
               comment.author.id = req.user._id;
               comment.author.username = req.user.username;
               //save comment
               comment.save();
               doctor.comments.push(comment);
               doctor.save();
               console.log(comment);
              
               res.redirect('/viewdoctor/' + doctor._id);
           }
        });
       }
   });
});

app.get("/viewdoctor/:id/comments/:commentId/edit",isLoggedIn,  function(req, res){
    // find doctor by id
            Comment.findById(req.params.commentId, function(err, comment){
                if(err)
                {
                    console.log(err);
                }
                else 
                {
                  res.render("edit-comments",{doctor_id: req.params.id, comment: comment});
                }
             });
});

app.put("/viewdoctor/:id/comments/:commentId", function(req, res){
   Comment.findByIdAndUpdate(req.params.commentId, req.body.comment, function(err, comment){
       if(err){
           res.render("edit");
       } else {
           res.redirect("/viewdoctor/" + req.params.id);
       }
   });
});

app.delete("/viewdoctor/:id/comments/:commentId", function(req, res){
    Comment.findByIdAndRemove(req.params.commentId, function(err){
        if(err){
            console.log("PROBLEM!");
        } else {
            res.redirect("/viewdoctor/" + req.params.id);
        }
    });
});


/*doctor.create( {
        name:"Harry Potter98",
        Qualification:"M.D",
        Specilaization:"marijuana",
        image:"https://images.pottermore.com/bxd3o8b291gf/3SQ3X2km8wkQIsQWa02yOY/25f258f21bdbe5f552a4419bb775f4f0/HarryPotter_WB_F4_HarryPotterMidshot_Promo_080615_Port.jpg?w=1200",
        Skypeid:"live:jmathias98",
        
    },
    function(err,doctor){if(err){
                             console.log(err);                         }
                             else{
                             console.log("doctor");
                           console.log(doctor);
                         }
    Comment.create(
        {
        text:"This is amazing",
        author:"len",
        
        },
        function(err,comment){
            if(err){
                             console.log(err);                         }
                             else{
            doctor.comments.push(comment);
            doctor.save();
            console.log("Created comment");                 
             console.log(comment);
                             }             
                             
        
    });
                         
                         
                         
});
*/
//********AUTH ROUTES*********

// "/signin" => signIn.ejs..........it is the sigin page for user signup
app.get("/signIn",function(req,res){
    res.render("signIn");
})

app.post("/signIn",function(req,res){
    var newUser = new User({username : req.body.username, 
        name: req.body.name,
    
    number:req.body.number,
    email: req.body.email,
    level: 0,
    exuno: "",
    exu: "",
    microno: "",
    micro: "",
    bloodno:"",
    blood: "",
 
    });
    
     var mailOptions = {
        from: 'j.mathjen@gmail.com',
        to: newUser.email,
        subject: 'Greetings From RetinoDoc!!',
        html: '<h4>Hello</h4>' + newUser.username + '<h4>Thanks for visiting us.</h4><h4>You have succesfully signed in. We hope you have a wonderful experience</h4>'
    };

    transporter.sendMail(mailOptions, function(error, info) {
        if (error) {
            console.log(error);
        }
        else {
            console.log('Email sent: ' + info.response);
        }
    });
   
    User.register(newUser, req.body.password,function(err, User){
        if(err){
            console.log(err);
            return res.render("signIn");
        }
        console.log("New User created:");
        console.log(User);
        passport.authenticate("local")(req,res, function(){
          res.redirect("/");
        });
        
        });
        
        
        
        
    });

// ==============Login========================= \\


// "/login" => login.ejs..........it is the login page for user to login
app.get("/login",function(req,res){
    
    
    res.render("login");
})

app.post('/login',
  passport.authenticate('local', { successRedirect: "/", failureRedirect: "/login"})
);

// admin
app.get("/adminlogin",function(req,res){
    res.render("adminLogin");
});
// app.get("/users:id",function(req,res){
//     User.findById(req.params.id,function(err,foundUser){
//       if(err){
//           req.flash("error","Something went wrong");
//           res.redirect("/");
//       } 
//       res.render("users/userprofile",{user:foundUser});
//     });
    
// });
app.post('/adminlogin', function(req,res){
    if(req.body.username ==="admin" && req.body.password==="admin123"){
        res.render("admin");
    }
    else{
        res.render("adminLogin");
    }
}
);


// ==============Logout======================== \\

app.get("/logout",function(req,res){
   req.logout();
   res.redirect("/");
});

function isLoggedIn(req,res,next){
    if(req.isAuthenticated()){
        return next();
        
    }
    res.redirect("/login");
}

// =============Search Logic================== \\

function escapeRegex(text){
    return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&");
}

//listen to port (AWS  ports)
app.listen(process.env.PORT,process.env.IP,function(){
    
     console.log("Server Started Sucessfully......!! ");
     console.log(process.env.PORT,process.env.IP);
});

//==================View Profile 13th march=========================\\

// app.get('/userprofile', 
// passport.authenticate('local', { successRedirect: "/", failureRedirect: "/login"})
// );

app.get("/home/:id",isLoggedIn,  function(req, res){
    // find user by id
            User.findById(req.params.id, function(err){
                if(err)
                {
                    console.log(err);
                }
                else 
                {
                  console.log("Success");
                  res.render("userprofile",{user:User});
                }
             });
});


app.get("/trial",function(req,res){
    
   res.render('http://18.222.175.240:5000/');
});

app.use(express.static('/environment/wtproj/environment/public'));
