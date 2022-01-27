const INTERVAL = 500
const d = new Date()
let time = null
let stopUpload = true


let start_camera = document.querySelector("#start-camera")
let video = document.querySelector("#video")
let start_capture = document.querySelector('#start-capture')
let captured_image = document.querySelector('#captured-image')

//start camera display
start_camera.addEventListener('click', async function(){
    if (navigator.mediaDevices.getUserMedia || navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
        getUserMedia({video: { video: true, audio: false }}, success_func, error_func);
    } else {
        alert('media access not supported');
    }
})

//for above function
function getUserMedia(constraints, success, error){
    if (navigator.mediaDevices.getUserMedia) {
        //最新的标准API
        navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
    } else if (navigator.webkitGetUserMedia) {
        //webkit核心浏览器
        navigator.webkitGetUserMedia(constraints, success, error)
    } else if (navigator.mozGetUserMedia) {
        //firfox浏览器
        navigator.mozGetUserMedia(constraints, success, error);
    } else if (navigator.getUserMedia) {
        //旧版API
        navigator.getUserMedia(constraints, success, error);
    }
}

//for above function
function success_func(stream) {
    //兼容webkit核心浏览器
    var CompatibleURL = window.URL || window.webkitURL;
    //将视频流设置为video元素的源
    // console.log(stream);
    //video.src = CompatibleURL.createObjectURL(stream);
    video.srcObject = stream;
    start_camera.style.display = "none"
    start_capture.style.display = "flex"
}

function error_func(error) {
    console.log("media access denied");
}

//to do
//this only captures a single image
start_capture.addEventListener('click',function(){
    //capture img from video
    captured_image.getContext('2d').drawImage(video,0,0,captured_image.width,captured_image.height);
    //get img data
    let imageData = captured_image.toDataURL("image/png")
    //convert img data into img source
    let imgStr = imageData.substr(22)
    //create new img file from img source
    let img = new Image();
    img.src = imgStr
    
    /*
    blah.src = imageData
    blah.width = blah.width< blah.height?blah.width:blah.height; 
    blah.height = blah.width< blah.height?blah.width:blah.height; 
    */

    //redraw image from the newly created file
    captured_image.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);
})

//decides whether to upload or wait for a certain time
//put this in the callback of the upload fetch request

document.getElementById("start").addEventListener("click",startContinuousUpload)
document.getElementById("stop").addEventListener("click",stopContinuousUpload)

function startContinuousUpload(){
    newUpload()
    stopUpload = false
}

function stopContinuousUpload(){
    time = null
    stopUpload = true
}

function newUpload() {
    if (time == null){
        time = d.getTime()
        //here goes the function to capture and upload a new image
    }else if (d.getTime() - time >= INTERVAL){
        //here goes the function to capture and upload a new image
    }else{
        setTimeout(captureAndUpload, INTERVAL - d.getTime() + time)
    }
}


//load src and convert to a File instance object
//work for any type of src, not only image src.
//return a promise that resolves with a File instance
function srcToFile(src, fileName, mimeType){
    return (fetch(src)
        .then(function(res){return res.arrayBuffer();})
        .then(function(buf){return new File([buf], fileName, {type:mimeType});})
    );
}

function captureAndUpload() {
    //capture img from video
    captured_image.getContext('2d').drawImage(video,0,0,captured_image.width,captured_image.height);
    //get img data
    let imageData = captured_image.toDataURL("image/png")
    //convert img data into img source
    let imgStr = imageData.substr(22)
    //create new img file from img source
    let img = new Image();
    img.src = imgStr
    //redraw image from the newly created file
    captured_image.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);

    srcToFile(img.src, "image.png", "image/png")
      .then( imgFile => {
        let formData = new FormData()
        formData.append('image', imgFile)
        console.log(imgFile)
        return {
          method: 'PUT',
          body: formData,
        }
      })
      .then( data => {
        return fetch(url,data)
      })
      .then(response => {
          return response.json()
        })
      .then( res => {
          console.log(res)

          //update_skeleton(res['I2L_joints'], I2L_skeleton);
          //update_skeleton(res['human36_joints'], human36_skeleton);
          //update_skeleton(res['Sem_joints'], Sem_skeleton); 
          //check whether have decided to stop uploading
          if(!stopUpload){
            newUpload()
          }
      })
      .then(console.log("succeed here!"))
      .catch(error => console.log(error))


/*
  let formData = new FormData()
    formData.append('image', imgFile)

    for (var key of formData.values()) {
      console.log(key)
    }

    let data = {
        method: 'PUT',
        body: formData,
    }
    
    fetch(url,data).then(response => {
        return response.json()
    }).then(data => {
        console.log(data)

        //update_skeleton(data['I2L_joints'], I2L_skeleton);
        //update_skeleton(data['human36_joints'], human36_skeleton);
        //update_skeleton(data['Sem_joints'], Sem_skeleton); 

        //check whether have decided to stop uploading
        if(!stopUpload){
            newUpload()
        }
    }).catch(error => console.log(error))
    return 

*/
}
