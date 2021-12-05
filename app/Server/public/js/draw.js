import * as THREE from 'https://cdn.skypack.dev/three@0.133.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.133.1/examples/jsm/controls/OrbitControls.js';

const PORT = 5000
const URL = '127.0.0.1'
var ngrok_url = window.location.href
const canvas = document.getElementById("3dCanvas")


document.getElementById("form-image").onchange = evt => {
  let image = document.getElementById("form-image").files[0]
  if (image) {
    blah.src = URL.createObjectURL(image)
    blah.width = blah.width< blah.height?blah.width:blah.height; 
    blah.height = blah.width< blah.height?blah.width:blah.height; 
  }
}
form.addEventListener("submit",function(event){

    event.preventDefault();
    //let url = `http://${URL}:${PORT}/imageUpload`
    let url = `${ngrok_url}imageUpload`
    window.alert("trying to upload image to" + url)

    let image = document.getElementById("form-image").files[0]
    console.log(image)
    if (image == null || image == ''){
        window.alert("You haven't choose a file yet")
        return
    }
    
    let formData = new FormData()
    formData.append('image',image)
    let data = {
        method: 'POST',
        body: formData
    }
    
  fetch(url, data).then(response => response.json())
        .then(
            data => {
            console.log(data);
            update_skeleton(data['I2L_joints'], I2L_skeleton);
            update_skeleton(data['human36_joints'], human36_skeleton);
            update_skeleton(data['Sem_joints'], Sem_skeleton); 
                }
            )
        .catch(error => console.log(error))

})
const human36_indices=  
[   0, 7, 
    7, 8,
    8, 9, 
    9, 10,
    8, 11, 
    11, 12, 
    12, 13, 
    8, 14, 
    14, 15, 
    15,16, 
    0, 1, 
    1, 2, 
    2, 3, 
    0, 4, 
    4, 5, 
    5, 6 ];

const I2L_indices = 
[   0,1,
    1,4,
    4,7,
    7,10,
    0,2,
    2,5,
    5,8,
    8,11,
    0,3,
    3,6,
    6,9,
    9,14,
    14,17,
    17,19,
    19,21,
    21,23,
    9,13,
    13,16,
    16,18,
    18,20,
    20,22,
    9,12,
    12,24,
    24,15,
    24,25,
    24,26,
    25,27,
    26,28,];

var init_position =
[   39.7642, 22.7078, 31.9892,
     
    40.6116, 26.0905, 34.2193,
     
    36.0548, 25.0538, 32.6827,
     
    41.1295, 18.4470, 30.9866,
     
    43.6070, 38.7190, 29.3077,
     
    27.6858, 37.2311, 29.8864,
     
    43.4302, 16.8571, 27.4887,
     
    41.2941, 52.8330, 35.5459,
     
    19.6125, 47.5865, 37.6749,
     
    44.3492, 16.9762, 25.8378,
     
    42.8460, 55.7746, 32.3185,
     
    17.7097, 50.6393, 34.5038,
     
    47.5938, 12.3236, 20.6534,
     
    48.9126, 14.3306, 23.9226,
     
    43.3275, 13.3207, 22.2242,
     
    48.8497, 12.5166, 18.4363,
     
    52.3104, 14.6594, 24.7424,
     
    40.1984, 12.6481, 20.9878,
     
    53.4274, 18.9680, 31.7843,
     
    31.0148, 15.4037, 24.1125,
     
    53.2553, 26.5973, 26.6768,
     
    32.6980, 23.6892, 20.0303,
     
    53.2801, 28.8286, 24.4359,
     
    33.6501, 26.5843, 17.9701,
     
    50.3077, 12.9904, 14.9635,
     
    51.4026, 11.1491, 15.3668,
     
    48.9452, 10.8628, 15.0181,
     
    51.7383,  9.9970, 18.3521,
     
    46.2133,  9.2510, 16.5609,
     ]


let camera, controls, scene, renderer,I2L_skeleton, human36_skeleton,Sem_skeleton;

// 2d ARRAY
function flatten_array(multi_dim_array){
    let  oneD_position = [];
    /*for(let i = 0; i < multi_dim_array[0].length; i++)
    {
        oneD_position = oneD_position.concat(multi_dim_array[0][i]);
    }*/
    for(let i = 0; i < multi_dim_array.length; i++)
    {
        oneD_position = oneD_position.concat(multi_dim_array[i]);
    }
    return oneD_position;
}

function update_skeleton(new_position,skeleton){
    //set Pelvis to the origin
    new_position = flatten_array(new_position);
    console.log(new_position);
    let Pelvis_index = 0;
    
    let Pelvis_x = new_position[3*Pelvis_index];
    let Pelvis_y = new_position[3*Pelvis_index+1];
    let Pelvis_z = new_position[3*Pelvis_index+2];
    for (let i=0; i<new_position.length; i+=3){

        new_position[i] -=  Pelvis_x;
        new_position[i+1] -=  Pelvis_y;
        new_position[i+2] -=  Pelvis_z;
    }   

    const geometry = skeleton.geometry;    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new_position,3));
    
    }


function create_skelton(position, indices){
    //set Pelvis to the origin
  if (position.length == 0) {

    const geometry = new THREE.BufferGeometry();    
    geometry.setIndex(indices);    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position,3));
    
    const material = new THREE.MeshBasicMaterial( { color: 0xFF69B4 } );
    let skeleton = new THREE.LineSegments(geometry, material);
    const quaternion = new THREE.Quaternion();
    //quaternion.setFromUnitVectors( spine_vecotor,new THREE.Vector3( 0, 1, 0 ) );
    quaternion.setFromAxisAngle( new THREE.Vector3( 1, 0, 0 ), Math.PI  );
    skeleton.applyQuaternion(quaternion);
    return skeleton;

    
  }
  
    let Pelvis_index = 0, chest_index = 3;

    let Pelvis_x = position[3 * Pelvis_index];
    let Pelvis_y = position[3 * Pelvis_index + 1];
    let Pelvis_z = position[3 * Pelvis_index + 2];
    for (let i = 0; i < position.length; i += 3) {

      position[i] -= Pelvis_x;
      position[i + 1] -= Pelvis_y;
      position[i + 2] -= Pelvis_z;
    }
  

    const geometry = new THREE.BufferGeometry();    
    geometry.setIndex(indices);    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position,3));
    
    const material = new THREE.MeshBasicMaterial( { color: 0xFF69B4 } );
    let skeleton = new THREE.LineSegments(geometry, material);
    
    let spine_vecotor = new THREE.Vector3( position[chest_index*3]-position[Pelvis_index*3], position[chest_index*3+1]-position[Pelvis_index*3+1], position[chest_index*3+2]-position[Pelvis_index*3+2] ).normalize();
    const quaternion = new THREE.Quaternion();
    //quaternion.setFromUnitVectors( spine_vecotor,new THREE.Vector3( 0, 1, 0 ) );
    quaternion.setFromAxisAngle( new THREE.Vector3( 1, 0, 0 ), Math.PI  );
    skeleton.applyQuaternion(quaternion);
    
    return skeleton;
    }
    


function init(){

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xeeeeee );
    scene.add( new THREE.GridHelper( 400, 10 ) );
    scene.add( new THREE.AxesHelper( 100 ) );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.set( 15, 20, 30 );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize( canvas.offsetWidth - 25, window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720 );

    canvas.setAttribute("height",window.innerHeight - 60 > 720 ? window.innerHeight - 60 : 720)
    canvas.setAttribute("width", window.innerWidth - 20)
    canvas.appendChild( renderer.domElement );

    controls = new OrbitControls( camera, renderer.domElement );
    window.addEventListener( 'resize', onWindowResize );

  I2L_skeleton = create_skelton(init_position, I2L_indices);
    scene.add(I2L_skeleton);
  human36_skeleton = create_skelton([], human36_indices);
  human36_skeleton.position.x = -40;
    scene.add(human36_skeleton);
  Sem_skeleton = create_skelton([], human36_indices);
  Sem_skeleton.position.x = 40;
    scene.add(Sem_skeleton);
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    canvas.setAttribute("height",window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720)
    
    renderer.setSize( canvas.offsetWidth - 6, window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720 );

}


function animate() {
	requestAnimationFrame( animate );

	renderer.render( scene, camera );
}



init();
animate();
