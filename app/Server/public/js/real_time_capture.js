const INTERVAL = 500
const DEFAULT_TIMESTAMP = 1 //
const d = new Date()
let time = null
let stopUpload = true
let action_record = []
let url = `${ngrok_url}realTimeUpload`
import { update_skeleton} from './draw.js'

let start_camera = document.querySelector("#start-camera")
let video = document.querySelector("#video")
let start_capture = document.querySelector('#start-capture')
let captured_image = document.querySelector('#captured-image')

let actionCounter = 0
let accuracyRecord = []
let one_click_record = []
let startingTime

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

function startContinuousUpload() {
    result.style.display = 'block';
    action_record = []
    actionCounter = 0
    accuracyRecord = []
    one_click_record = []
    newUpload()
    startingTime = Date.now()
    stopUpload = false
}

function stopContinuousUpload() {
    result.style.display = 'none';
    time = null
    stopUpload = true
    console.log(action_record)

}

function newUpload() {
    setTimeout(test_only, INTERVAL )
    //setTimeout(captureAndUpload, INTERVAL )
    
    /*
    if (time == null){
        time = d.getTime()
        //here goes the function to capture and upload a new image
    }else if (d.getTime() - time >= INTERVAL){
        //here goes the function to capture and upload a new image
        stopContinuousUpload()
    }else{
        setTimeout(captureAndUpload, INTERVAL - d.getTime() + time)
    }
    */
    
}

// use static image to simulate real-time process 
function test_only() {
  let image = document.getElementById("form-image").files[0]
  let formData = new FormData()
  formData.append('image', image)
  formData.append('action_choice', document.getElementById("Action_Choice").value)
  let timestamp = (Date.now() - startingTime)/1000
  console.log(`timestamp is ${timestamp}`)
  formData.append('timestamp',timestamp)
  let data = {
    method: 'PUT',
    body: formData
  }
  fetch(url, data).then(response => {
    return response.json()
  })
    .then(res => {
      console.log(res)
      update_skeleton(res['smpl_joint_coords'], I2L_skeleton);

        if (res['action_name'] != 'Loss exceeds threshold!') {
          action_record.push({
            'action_name': res['action_name'],
            'loss': res['loss']
          })
          // need to improve for a batch of image upload
          one_click_record.push({
            'smpl_coord': res['smpl_joint_coords'],
            'timestamp': timestamp

          })
        }
          //update_skeleton(res['Sem_joints'], Sem_skeleton); 
      // action recognition
      let dict = new Object()
      let max_value = 0.00001
      let predicted_action = ''
      let total_loss = 0
      // need to extract data from res
      let replay_data = one_click_record;
      accuracyRecord.push(res.action_accuracy)
        if (action_record.length > 10) {

            for (let i = 2; i < action_record.length-2; i++) {
              if (action_record[i]['action_name'] in dict) {
                dict[action_record[i]['action_name']] += 1
                if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
              else {
                dict[action_record[i]['action_name']] = 1
                  if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
            }
          console.log ('predicted action is:'+ predicted_action)
          Action_Name.text = predicted_action

          for (let i = 0; i < action_record.length; i++) {
            if (action_record[i]['action_name'] == predicted_action)
              total_loss += action_record[i]['loss']
          }
          Average_Score.text = total_loss/max_value
          actionCounter++
      }
      if (!stopUpload)
        newUpload()
      else {
        let tr = document.createElement("tr")
			  let td_1 = document.createElement("td")
			  td_1.appendChild(document.createTextNode(predicted_action))
			  let td_2 = document.createElement("td")
			  td_2.appendChild(document.createTextNode(total_loss/max_value))
			  let td_3 = document.createElement("td")
        td_3.appendChild(document.createTextNode('NA'))
        let td_4 = document.createElement("td");
			  let butt = document.createElement("button");
        td_4.appendChild(butt);
        butt.className = "btn btn-primary"
			  // use function closure to bind the block scpoed record_data to current one 
			  butt.addEventListener('click', ()=>{
				  console.log(replay_data)
			  });
			  tr.appendChild(td_1)
			  tr.appendChild(td_2)
        tr.appendChild(td_3)
        tr.appendChild(td_4);
        record.append(tr)
        Action_Name.text = ''
        Average_Score.text = ''
        console.log(accuracyRecord)
      }

    })
    .then(console.log("test succeed"))
    .catch(err => console.log(err))
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

function dataURItoBlob(dataURI) {
    // convert base64 to raw binary data held in a string
    // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
    var byteString = atob(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    //New Code
    return new Blob([ab], {type: mimeString});

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

    let imgBlob = dataURItoBlob(imageData)
    let imgFile = new File([imgBlob], "image.png");
    console.log(imgFile)
    let formData = new FormData()
    formData.append('image', imgFile)
    formData.append('action_choice', document.getElementById("Action_Choice").value)
    formData.append('timestamp',DEFAULT_TIMESTAMP)
    let data = {
          method: 'PUT',
          body: formData,
        }

      fetch(url,data).then(response => {
          return response.json()
        })
      .then( res => {
          console.log(res)

          update_skeleton(res['smpl_joint_coords'], I2L_skeleton);
          //update_skeleton(res['human36_joint_coords'], human36_skeleton);


        if (action_record[0]['action_name'] != 'Loss exceeds threshold!') {
          action_record.push({
            'action_name': res['action_name'],
            'loss': res['loss']
          })                
        }


          //update_skeleton(res['Sem_joints'], Sem_skeleton); 
        // action recognition
        let dict = new Object()
        let max_value = 0.00001
        let predicted_action = ''
        let total_loss = 0
        // need to extract data from res
        let replay_data = res;
        if (action_record.length > 10) {

          for (let i = 2; i < action_record.length-2; i++) {
            if (action_record[i]['action_name'] in dict) {
              dict[action_record[i]['action_name']] += 1
              if (dict[action_record[i]['action_name']] > max_value) {
                max_value = dict[action_record[i]['action_name']]
                predicted_action = action_record[i]['action_name']
              }
            }
              else {
                dict[action_record[i]['action_name']] = 1
                if (dict[action_record[i]['action_name']] > max_value) {
                  max_value = dict[action_record[i]['action_name']]
                  predicted_action = action_record[i]['action_name']
                }
              }
            }
          console.log ('predicted action is:'+ predicted_action)
          Action_Name.text = predicted_action

          for (let i = 0; i < action_record.length; i++) {
            if (action_record[i]['action_name'] == predicted_action)
              total_loss += action_record[i]['loss']
          }
          Average_Score.text = total_loss/max_value
        }
          

        //check whether have decided to stop uploading
        if(!stopUpload){
          newUpload()
        }
        else {
          let tr = document.createElement("tr");
          let td_1 = document.createElement("td");
          td_1.appendChild(document.createTextNode(predicted_action));
          let td_2 = document.createElement("td");
          td_2.appendChild(document.createTextNode(total_loss / max_value));
          let td_3 = document.createElement("td");
          td_3.appendChild(document.createTextNode('NA'));
          let td_4 = document.createElement("td");
			    let butt = document.createElement("button");
          td_4.appendChild(butt);
          butt.className = "btn btn-primary"
			    // use function closure to bind the block scpoed record_data to current one 
			    butt.addEventListener('click', ()=>{
				    console.log(replay_data)
			    });
          tr.appendChild(td_1);
          tr.appendChild(td_2);
          tr.appendChild(td_3);
          tr.appendChild(td_4);
          record.append(tr);
          Action_Name.text = ''
          Average_Score.text = ''
      }
      })
      .then(console.log("succeed here!"))
      .catch(error => console.log(error))

}
