const wrapper = document.querySelector('.wrapper');
const loginLink = document.querySelector('.login-link');
const registerLink = document.querySelector('.register-link');
const btnPopup = document.querySelector('#btn');
const iconClose = document.querySelector('.icon-close');
const desSection = document.querySelector('.des');

registerLink.addEventListener('click', () =>{
    wrapper.classList.add('active');
    wrapper.style.display = 'block';
    desSection.style.display = 'none';
    document.getElementById('loginForm').reset();
})
loginLink.addEventListener('click', () =>{
    wrapper.classList.remove('active');
    wrapper.style.display = 'block';
    desSection.style.display = 'none';
    document.getElementById('registerForm').reset();
})
btnPopup.addEventListener('click', () =>{
    wrapper.classList.add('active-popup');
    wrapper.style.display = 'block';
    desSection.style.display = 'none';
})
iconClose.addEventListener('click', () =>{
    wrapper.classList.remove('active-popup');
    wrapper.style.display = 'none';
    desSection.style.display = 'block';
    document.getElementById('loginForm').reset();
    document.getElementById('registerForm').reset();
})
