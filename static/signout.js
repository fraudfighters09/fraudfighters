let UserCreds = JSON.parse(sessionStorage.getItem("user-creds"));
let UserInfo = JSON.parse(sessionStorage.getItem("user-info"));

let signOut = document.getElementById('signout');
let Signout = () =>{
    sessionStorage.removeItem("user-creds");
    sessionStorage.removeItem("user-info");
    showToast("Signed Out Successfully...","success",5000);
    setTimeout(() => { 
           window.location.href = '/';
    }, 3000); 
}
let CheckCred = () => {
    if(!sessionStorage.getItem("user-creds"))
       window.location.href = '/';
}
window.addEventListener('load',CheckCred);
signOut.addEventListener('click', Signout);