import React, { useState, useEffect } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import 'react-time-picker/dist/TimePicker.css';
import { formatDate } from '../utils/dateUtils';

const PredictForm = () => {
    const [params, setParams] = useState({
        model_id: '',
        start_date: '',
        end_date: ''
    });
    const [result, setResult] = useState(null);
    const [selectedModelId, setSelectedModelId] = useState('');
    const [selectedDate, setSelectedDate] = useState('');
    const [filteredConfigurations, setFilteredConfigurations] = useState([]);
    const [configurations, setConfigurations] = useState([]);
    const [modelIds, setModelIds] = useState([]);
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date());

    const handleChange = (e) => {
        setParams({ ...params, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const formattedStartDate = formatDate(startDate);
        const formattedEndDate = formatDate(endDate);

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(
                {
                    model_id: selectedModelId,
                    start_date: formattedStartDate,
                    end_date: formattedEndDate,
                    configuration_date: selectedDate
                }
            )
        });
        const data = await response.json();
        setResult(data);
    };

    useEffect(() => {
        fetchConfigurations();
    }, []);

    const fetchConfigurations = async () => {
        try {
            const response = await axios.get('/api/training/retrieve/configurations');
            console.log(response.data.configurations);
            setConfigurations(response.data.configurations);
            setModelIds([...new Set(response.data.configurations.map(item => item.model_id))]);
        } catch (error) {
            console.error('Error fetching configurations:', error);
        }
    };

    const handleModelChange = (event) => {
        const modelId = event.target.value;
        setSelectedModelId(modelId);
        filterConfigurations(selectedDate, modelId);
    };

    const handleDateChange = (event) => {
        const date = event.target.value;
        console.log(date)
        setSelectedDate(date);
        filterConfigurations(date, selectedModelId);
    };

    const filterConfigurations = (date, modelId) => {
        if (date && modelId) {
            const filtered = configurations.filter(config => 
                config.timestamp === date && String(config.model_id) === String(modelId)
            );
            console.log(filtered);
            setFilteredConfigurations(filtered);
        } else {
            setFilteredConfigurations([]);
        }
    };

    return (
        <div>
            <h2>Predict</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Select Model ID: </label>
                    <select value={selectedModelId} onChange={handleModelChange}>
                        <option value="">Select a model ID</option>
                        {[...new Set(configurations.map(config => config.model_id))].map(id => (
                            <option key={id} value={id}>{id}</option>
                        ))}
                    </select>
                </div>
                <div>
                    <label>Select Configuration Date: </label>
                    <select value={selectedDate} onChange={handleDateChange}>
                        <option value="">Select a date</option>
                        {[...new Set(configurations.map(config => config.timestamp))].map(date => (
                            <option key={date} value={date}>{date}</option>
                        ))}
                    </select>
                </div>
                <div className="date-time-picker-container">
                    <div className="date-time-picker">
                        <label>Start Date and Time:</label>
                        <DatePicker
                            selected={startDate}
                            onChange={date => setStartDate(date)}
                            showTimeSelect
                            dateFormat="Pp"
                        />
                    </div>

                    <div className="date-time-picker">
                        <label>End Date and Time:</label>
                        <DatePicker
                            selected={endDate}
                            onChange={date => setEndDate(date)}
                            showTimeSelect
                            dateFormat="Pp"
                        />
                    </div>
                </div>
                <button 
                    className="block text-lg py-3 px-4 w-full border outline-none focus:ring focus:outline-none rounded-pill"
                    style={{ fontSize: '1.25rem' }}
                    type="submit"
                >Predict</button>
            </form>
            {result && <div>Execution Time: {result.execution_time}s</div>}
        </div>
    );
};

export default PredictForm;