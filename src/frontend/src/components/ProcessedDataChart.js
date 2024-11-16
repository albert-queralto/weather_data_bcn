import React, { useEffect, useState } from 'react';
import axios from 'axios';
import * as am5 from "@amcharts/amcharts5";
import * as am5xy from "@amcharts/amcharts5/xy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../styles/Chart.css';

const ChartComponent = () => {
  const [data, setData] = useState([]);
  const [latitude, setLatitude] = useState("41.389");
  const [longitude, setLongitude] = useState("2.159");
  const [startDate, setStartDate] = useState(new Date('2024-09-01'));
  const [endDate, setEndDate] = useState(new Date('2024-10-02'));
  const [selectedVariables, setSelectedVariables] = useState([]);

  useEffect(() => {
    axios.post('http://localhost:8100/get_data', {
      latitude,
      longitude,
      start_date: startDate.toISOString().split('T')[0],
      end_date: endDate.toISOString().split('T')[0]
    })
    .then(response => {
      setData(response.data.data);
    })
    .catch(error => {
      console.error('There was an error fetching the data!', error);
    });
  }, [latitude, longitude, startDate, endDate]);

  useEffect(() => {
    if (data.length === 0) return;

    const groupedData = data.reduce((acc, item) => {
      if (!acc[item.variable_code]) {
        acc[item.variable_code] = [];
      }
      acc[item.variable_code].push({
        date: new Date(item.timestamp).getTime(),
        value: item.value,
        variable_code: item.variable_code // Include variable_code in the data
      });
      return acc;
    }, {});

    console.log("Grouped Data:", groupedData);

    let root = am5.Root.new("chartdiv");
    root.setThemes([am5themes_Animated.new(root)]);

    let chart = root.container.children.push(am5xy.XYChart.new(root, {
      panX: true,
      panY: true,
      wheelX: "panX",
      wheelY: "zoomX",
      pinchZoomX: true
    }));

    let xAxis = chart.xAxes.push(am5xy.DateAxis.new(root, {
      maxDeviation: 0.2,
      baseInterval: {
        timeUnit: "hour",
        count: 1
      },
      renderer: am5xy.AxisRendererX.new(root, {}),
      tooltip: am5.Tooltip.new(root, {}),
      dateFormats: {
        hour: "HH:mm"
      }
    }));

    let yAxis = chart.yAxes.push(am5xy.ValueAxis.new(root, {
      renderer: am5xy.AxisRendererY.new(root, {})
    }));

    Object.keys(groupedData).forEach(variable_code => {
      if (selectedVariables.length === 0 || selectedVariables.includes(variable_code)) {
        let series = chart.series.push(am5xy.LineSeries.new(root, {
          name: variable_code,
          xAxis: xAxis,
          yAxis: yAxis,
          valueYField: "value",
          valueXField: "date",
          strokeWidth: 2,
          tooltip: am5.Tooltip.new(root, {
            labelText: "{name}: {valueY}"
          })
        }));

        series.data.setAll(groupedData[variable_code]);

        series.bullets.push(function () {
          let bulletCircle = am5.Circle.new(root, {
            radius: 5,
            fill: series.get("fill")
          });
          return am5.Bullet.new(root, {
            sprite: bulletCircle
          });
        });
      }
    });

    chart.set("cursor", am5xy.XYCursor.new(root, {}));
    chart.set("scrollbarX", am5.Scrollbar.new(root, {
      orientation: "horizontal"
    }));

    return () => {
      root.dispose();
    };
  }, [data, selectedVariables]);

  const handleMapClick = (e) => {
    setLatitude(e.latlng.lat);
    setLongitude(e.latlng.lng);
  };

  const handleVariableChange = (event) => {
    const { options } = event.target;
    const selected = [];
    for (const option of options) {
      if (option.selected) {
        selected.push(option.value);
      }
    }
    setSelectedVariables(selected);
  };

  const uniqueVariableCodes = [...new Set(data.map(item => item.variable_code))];

  return (
    <div className="container">
      <div className="sidebar">
        <div className="form-group">
          <label>Start Date: </label>
          <DatePicker 
            selected={startDate} 
            onChange={date => setStartDate(date)} 
            showTimeSelect
            dateFormat="Pp"
            popperClassName="datepicker-container"
          />
        </div>
        <div className="form-group">
          <label>End Date: </label>
          <DatePicker 
            selected={endDate} 
            onChange={date => setEndDate(date)} 
            showTimeSelect
            dateFormat="Pp"
            popperClassName="datepicker-container"
          />
        </div>
        <div className="form-group">
          <label>Variable Codes: </label>
          <select multiple={true} value={selectedVariables} onChange={handleVariableChange}>
            {uniqueVariableCodes.map(code => (
              <option key={code} value={code}>{code}</option>
            ))}
          </select>
        </div>
        <div className="map-container">
          <MapContainer center={[latitude, longitude]} zoom={13} style={{ height: "200px", width: "100%" }} onClick={handleMapClick}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </MapContainer>
        </div>
      </div>
      <div className="main-content">
        <div className="chart-container">
          <div id="chartdiv" style={{ width: "100%", height: "400px" }}></div>
        </div>
      </div>
    </div>
  );
};

export default ChartComponent;