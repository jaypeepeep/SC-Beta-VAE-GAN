// Get the canvas element and its context
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');

// Variables to keep track of drawing state
let isDrawing = false;
let hideInAir = false; // Flag to control visibility of in-air drawings

// Array to store drawing data
const drawingData = [];

// Set up the initial drawing properties
ctx.strokeStyle = 'black'; // Default drawing color
ctx.lineWidth = 2; // Width of the drawing line
ctx.lineJoin = 'round'; // Smooth line joins
ctx.lineCap = 'round'; // Smooth line caps

// Scaling factor to amplify pixel coordinates
const scalingFactor = 100; // Adjust this factor to achieve the desired range

// Maximum value for pressure (based on reference values)
const maxPressure = 255; // Adjust this value if needed

// Function to convert altitude from radians to a scaled whole number
function convertAltitude(radians) {
    const degrees = radians * (180 / Math.PI); // Convert radians to degrees
    const scaledAltitude = Math.round(degrees * 10); // Scale and round to achieve a value similar to 3 digits
    return scaledAltitude;
}

// Function to convert pressure to a scaled whole number
function convertPressure(pressure) {
    // Assuming pressure is normalized between 0 and 1
    const scaledPressure = Math.round(pressure * maxPressure); // Scale and round to get whole number
    return Math.min(Math.max(scaledPressure, 0), maxPressure); // Ensure pressure is within valid range
}

// Function to compute azimuth based on altitude
function computeAzimuth() {
    // Define a base value and range for variability
    const baseValue = 1900;
    const range = 100; // Range of variability

    // Generate azimuth value with some variability
    const azimuth = baseValue + Math.floor(Math.random() * range) - (range / 2);

    // Round to the nearest 10
    return Math.round(azimuth / 10) * 10;
}

// Function to handle drawing and recording data
function draw(x, y, pressure, azimuth, altitude) {
    const timestamp = Math.floor(performance.now() * 1000); // Timestamp in microseconds
    const penStatus = isDrawing ? 1 : 0;

    const scaledX = Math.round(x * scalingFactor);
    const scaledY = Math.round(y * scalingFactor);
    const scaledAltitude = convertAltitude(altitude); // Convert and scale altitude
    const scaledPressure = convertPressure(pressure); // Convert and scale pressure
    const computedAzimuth = computeAzimuth(); // Compute azimuth

    if (isDrawing) {
        ctx.lineTo(x, y);
        ctx.stroke();
    } else {
        if (!hideInAir) {
            ctx.strokeStyle = 'red'; // Change color to red when hovering
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x, y);
            ctx.stroke();
        }
    }

    // Store the data
    addDrawingData(scaledX, scaledY, timestamp, penStatus, computedAzimuth, scaledAltitude, scaledPressure);
}

// Function to add data to the array
function addDrawingData(x, y, timestamp, penStatus, azimuth, altitude, pressure) {
    drawingData.push([x, y, timestamp, penStatus, azimuth, altitude, pressure]);
}

// Function to convert array data to SVC string
function convertToSVC(data) {
    const header = 'x y timestamp penStatus azimuth altitude pressure\n';
    const rows = data.map(row => row.join(' ')).join('\n');
    return header + rows;
}

// Function to download SVC file
function downloadSVC() {
    const svcData = convertToSVC(drawingData);
    const blob = new Blob([svcData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'drawingData.svc';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Function to clear the canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawingData.length = 0;

}

// Add event listener for downloading the SVC file
document.getElementById('downloadButton').addEventListener('click', downloadSVC);

// Add event listener for downloading the SVC file
document.getElementById('resetButton').addEventListener('click', clearCanvas);

// Add event listener for checkbox to toggle in-air drawing visibility and clear canvas
document.getElementById('hideInAirCheckbox').addEventListener('change', (e) => {
    hideInAir = e.target.checked;
    clearCanvas(); // Clear the canvas when the checkbox is clicked
});

// Prevent default touch actions like scrolling
function preventDefault(e) {
    e.preventDefault();
}

// Start drawing when the pen is down or mouse is pressed
canvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    ctx.strokeStyle = 'black';
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Stop drawing when the pen is up or mouse is released
canvas.addEventListener('pointerup', () => {
    isDrawing = false;
    ctx.beginPath(); // Reset the path
});

// Draw on the canvas while the pen is moving
canvas.addEventListener('pointermove', (e) => {
    const x = e.offsetX;
    const y = e.offsetY;
    const pressure = e.pressure || 0; // Default pressure to 0 if not available
    const azimuth = e.azimuthAngle || 0; // Default azimuth to 0 if not available
    const altitude = e.altitudeAngle || 0; // Default altitude to 0 if not available
    draw(x, y, pressure, azimuth, altitude);
});

// Event listener for the Done button to clear layout and reset canvas
document.getElementById('doneButton').addEventListener('click', () => {
    // Attempt to close the current tab
    window.close();
});


// Prevent default actions for touch events to avoid scrolling
canvas.addEventListener('touchstart', preventDefault, { passive: false });
canvas.addEventListener('touchmove', preventDefault, { passive: false });
canvas.addEventListener('touchend', preventDefault, { passive: false });
