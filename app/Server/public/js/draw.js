import * as THREE from 'https://cdn.skypack.dev/three@0.133.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.133.1/examples/jsm/controls/OrbitControls.js';

const PORT = 5000
const URL = '127.0.0.1'
const ngrok_url = 'http://1a0c-35-222-46-113.ngrok.io'
const canvas = document.getElementById("3dCanvas");

function uploadImage() {

}

form.addEventListener("submit",function(event){

    event.preventDefault();
    //let url = `http://${URL}:${PORT}/imageUpload`
    let url = `${ngrok_url}/imageUpload`
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
            console.log(data['coordinates']);
            update_skeleton(data['coordinates']); 
                }
            )
        .catch(error => console.log(error))

})

const indices = 
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

var position =
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

let possible_new_position = [[32.5407, 31.5434, 32.0625],
[32.0831, 31.5208, 32.6997],
[32.8597, 31.7833, 32.8044],
[33.7914, 31.2277, 31.6434],
[31.3528, 33.2767, 35.0507],
[29.4545, 33.3968, 27.2765],
[34.6095, 31.0597, 30.3914],
[32.6895, 36.0617, 41.2097],
[31.2094, 36.8942, 43.9023],
[34.3237, 31.4657, 31.1524],
[31.3542, 33.0677, 45.9145],
[31.8602, 33.6842, 39.9711],
[36.2314, 32.1091, 29.8994],
[34.8465, 30.9072, 29.8246],
[36.3953, 32.6197, 30.3386],
[35.1741, 29.5423, 31.1576],
[34.6915, 30.3615, 32.2895],
[36.0089, 32.2513, 32.0608],
[31.3736, 33.2176, 31.3728],
[30.9953, 34.5236, 33.4797],
[26.0312, 33.2727, 30.1604],
[22.2192, 33.6117, 30.6902],
[23.9756, 31.2753, 28.5218],
[17.2580, 34.9558, 34.4542],
[32.8368, 28.9838, 23.8058],
[32.8343, 27.8043, 22.3165],
[33.0859, 30.7670, 29.7692],
[33.2941, 28.6702, 26.0253],
[35.1356, 32.5765, 29.6174]]


let camera, controls, scene, renderer,skeleton;

function flatten_array(multi_dim_array){
    let  oneD_position = [];
    for(let i = 0; i < multi_dim_array.length; i++)
    {
        oneD_position = oneD_position.concat(multi_dim_array[i]);
    }
    return oneD_position;
}

function update_skeleton(new_position){
    //set Pelvis to the origin
    new_position = flatten_array(new_position);
    let Pelvis_index = 0, chest_index  = 3;
    
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


function create_skelton(position){
    //set Pelvis to the origin
    let Pelvis_index = 0, chest_index  = 3;

    let Pelvis_x = position[3*Pelvis_index];
    let Pelvis_y = position[3*Pelvis_index+1];
    let Pelvis_z = position[3*Pelvis_index+2];
    for (let i=0; i<position.length; i+=3){

        position[i] -=  Pelvis_x;
        position[i+1] -=  Pelvis_y;
        position[i+2] -=  Pelvis_z;
    }   

    const geometry = new THREE.BufferGeometry();    
    geometry.setIndex(indices);    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position,3));
    
    const material = new THREE.MeshBasicMaterial( { color: 0xFF69B4 } );
    skeleton = new THREE.LineSegments(geometry, material);
    
    let spine_vecotor = new THREE.Vector3( position[chest_index*3]-position[Pelvis_index*3], position[chest_index*3+1]-position[Pelvis_index*3+1], position[chest_index*3+2]-position[Pelvis_index*3+2] ).normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors( spine_vecotor,new THREE.Vector3( 0, 1, 0 ) );
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

    const skeleton = create_skelton(position);
    scene.add( skeleton );
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    canvas.setAttribute("height",window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720)
    
    renderer.setSize( canvas.offsetWidth - 6, window.innerHeight - 60  > 720 ? window.innerHeight - 60  : 720 );

}


function animate() {
	requestAnimationFrame( animate );

    update_skeleton(oneD_position);
	renderer.render( scene, camera );
}



init();
animate();
